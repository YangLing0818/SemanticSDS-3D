import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod

class AbstractAutoencoder(nn.Module, ABC):
    def __init__(self):
        super(AbstractAutoencoder, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass # NOTE returns logits

    @abstractmethod
    def semantic_BHWC_to_onehot_with_background(self, x):
        """
        x: (batchsize, H, W, encoding_dim)
        return onehots: (batchsize, H, W, num_classes)
        """
        pass

    @abstractmethod
    def groupids_to_semantics(self, groupids):
        """
        groupids: (N,) dtype=torch.int32
        return semantics: (N, 3)
        """
        pass

    @abstractmethod
    def get_bg4semantics(self, H, W):
        """
        H, W: int
        return bg: (H, W, 3)
        """
        pass

class OnehotAutoencoder(AbstractAutoencoder):
    def __init__(self, num_classes, encoding_dim=3, device=None):
        assert device is not None
        super(OnehotAutoencoder, self).__init__()
        self.num_classes = num_classes
        self.encoding_dim = encoding_dim
        self.device = device
        self.encoder = nn.Linear(num_classes, encoding_dim).to(device)
        self.decoder = nn.Linear(encoding_dim, num_classes).to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded # decoded is logits
    
    def semantic_BHWC_to_onehot_with_background(self, x):
        """
        x: (batchsize, H, W, encoding_dim)
        onehots: (batchsize, H, W, num_classes)
        """
        batchsize, H, W, _ = x.shape
        
        x = x.reshape(batchsize * H * W, self.encoding_dim)
        logits = self.decoder(x).reshape(batchsize, H, W, self.num_classes)
        class_ids = logits.argmax(dim=-1) # torch.Size([4, 512, 512])
        onehots = torch.nn.functional.one_hot(
            class_ids, 
            num_classes=self.num_classes
            ).float()
        
        return onehots # torch.Size([4, 512, 512, num_classes])
    
    def groupids_to_semantics(self, groupids):
        """
        groupids: (N,) dtype=torch.int32
        semantics: (N, 3)
        """
        onehots = torch.nn.functional.one_hot(
            groupids.to(torch.int64), 
            num_classes=self.num_classes
            ).to(torch.float32) # shape (self.N, self.num_classes)
        semantics = self.encoder(onehots) # shape (self.N, 3)
        return semantics
    
    def get_bg4semantics(self, H, W):
        """
        H, W: int
        return bg: (H, W, 3)
        """
        onehots = torch.zeros([1, self.num_classes], device=self.device, dtype=torch.float32) # torch.Size([1, 3])
        onehots[:, -1] = 1.0
        semantics = self.encoder(onehots)
        # semantics.shape == (1, 3)
        # repeat semantics to match (H, W, 3)
        semantics = semantics.repeat(H, W, 1) # ->@torch.Size([512, 512, 3])
        return semantics
    
    
class CLIPAutoencoder(nn.Module):
    def __init__(self, 
                    high_dim_embeds,
                    encoding_dim=3, 
                ):
        """
        [input] high_dim_embeds: (N, 512)
        self.bg_high_dim: (512)
        self.high_dim_embeds_with_bg: (N+1, 512)
        """
        super(CLIPAutoencoder, self).__init__()
        self.num_classes = high_dim_embeds.shape[0] + 1 # +1 for background
        self.encoding_dim = encoding_dim
        self.device = high_dim_embeds.device

        # Use register_buffer for tensors
        self.register_buffer('bg_high_dim', torch.randn(high_dim_embeds.shape[1]))
        self.register_buffer('original_high_dim_embeds_with_bg', 
                             torch.cat([high_dim_embeds, self.bg_high_dim.unsqueeze(0)], dim=0))
        self.register_buffer('noise', torch.randn_like(self.original_high_dim_embeds_with_bg) * 0.1)
        self.register_buffer('high_dim_embeds_with_bg', self.original_high_dim_embeds_with_bg + self.noise)

        self.encoder = nn.Linear(512, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, 512)

        # Move the entire module to the specified device
        self.to(self.device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def get_low_dim_embeds(self):
        if not hasattr(self, '_low_dim_embeds'):
            self.register_buffer('_low_dim_embeds', self.encode(self.high_dim_embeds_with_bg))
        if self._low_dim_embeds.device != self.device:
            self._low_dim_embeds = self._low_dim_embeds.to(self.device)
        return self._low_dim_embeds

    @property
    def low_dim_embeds(self): # shape==(N+1, 3)
        return self.get_low_dim_embeds()
    
    @property
    def bg_low_dim(self): # shape == (3)
        return self.low_dim_embeds[-1]
    
    @staticmethod
    def cal_clip_loss(predicted_embeds, high_dim_embeds):
        # contrastive loss function, adapted from
        # https://sachinruk.github.io/blog/2021-03-07-clip.html
        def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
            return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

        def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
            caption_loss = contrastive_loss(similarity)
            image_loss = contrastive_loss(similarity.t())
            return (caption_loss + image_loss) / 2.0
        
        
        predicted_norm = torch.nn.functional.normalize(predicted_embeds, p=2, dim=-1)
        high_dim_norm = torch.nn.functional.normalize(high_dim_embeds, p=2, dim=-1)

        logit_scale = torch.tensor([2.6593], device=predicted_norm.device).exp()
        logits_per_prompt = torch.matmul(predicted_norm, high_dim_norm.t()) * logit_scale
        loss = clip_loss(logits_per_prompt)
        return loss
    
    def classify(self, low_dim_embeds):
        """
        low_dim_embeds: (N, 3)
        return: {
            "class_ids": (N,),
            "classify_probability": (N,),
            "logits_per_prompt": (N, N),
        }
        """
        low_dim_embeds = low_dim_embeds.to(self.device)
        
        decoded = self.decode(low_dim_embeds)
        
        decoded_norm = torch.nn.functional.normalize(decoded, p=2, dim=-1)
        high_dim_norm = torch.nn.functional.normalize(self.high_dim_embeds_with_bg, p=2, dim=-1)

        logit_scale = torch.tensor([2.6593], device=self.device).exp()
        logits_per_prompt = torch.matmul(decoded_norm, high_dim_norm.t()) * logit_scale

        probabilities_per_prompt = logits_per_prompt.softmax(dim=-1)
        
        classify_probability, class_ids = probabilities_per_prompt.max(dim=1)

        return {
            "class_ids": class_ids,
            "classify_probability": classify_probability,
            "logits_per_prompt": logits_per_prompt,
        }
        
    def self_check(self):
        """
        Perform a self-check to verify if all self.classify(self.low_dim_embeds)["class_ids"] 
        are (0, 1, 2, ..., self.num_classes - 1).
        
        Returns:
        - bool: True if the check passes, False otherwise.
        - str: A message describing the result of the check.
        """
        with torch.no_grad():
            classification_result = self.classify(self.low_dim_embeds)
            class_ids = classification_result["class_ids"]
            
            expected_class_ids = torch.arange(self.num_classes, device=self.device)
            
            if torch.all(class_ids == expected_class_ids):
                return True
            else:
                return False
        
    def semantic_BHWC_to_onehot_with_background(self, x):
        """
        x: (batchsize, H, W, encoding_dim)
        [returned] onehots: (batchsize, H, W, num_classes)
        """
        batchsize, H, W, _ = x.shape
        
        x = x.reshape(batchsize * H * W, self.encoding_dim)
        classification_result = self.classify(x)
        class_ids = classification_result["class_ids"]
        
        num_classes = len(self.high_dim_embeds_with_bg)
        onehots = torch.nn.functional.one_hot(class_ids, num_classes=num_classes).float()
        onehots = onehots.reshape(batchsize, H, W, num_classes)
        
        return onehots
    
    def groupids_to_semantics(self, groupids):
        """
        groupids: (N,) dtype=torch.int32
        [return] semantics: (N, 3)
        """
        # self.low_dim_embeds.shape == (N+1, 3)
        semantics = self.low_dim_embeds[groupids]
        return semantics
    
    def get_bg4semantics(self, H, W):
        """
        H, W: int
        [return] bg: (H, W, 3)
        """
        # self.bg_low_dim.shape == (3)
        bg = self.bg_low_dim.repeat(H, W, 1)
        return bg
