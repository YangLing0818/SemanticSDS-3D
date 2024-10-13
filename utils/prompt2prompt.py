# Based on https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_prompt2prompt.py
from diffusers.models.attention import Attention
import abc
import torch
import numpy as np
import torch.nn.functional as F
from typing import List


class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape # (4, 4096, 320)
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query) # (32, 4096, 40) (batch_size * heads, seq_len, dim // heads)
        key = attn.head_to_batch_dim(key) # (32, 77, 40). self_attention时为(32, 4096, 40)...
        value = attn.head_to_batch_dim(value) # (32, 77, 40)

        attention_probs = attn.get_attention_scores(query, key, attention_mask) # (32, 4096, 77)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet) # AttentionControl.__call__

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    

class AttentionControl(abc.ABC):
    def step_callback(self, x_t): 
        return x_t
        # what's the difference between step_callback and between_steps?
        # Below is copied from google's implementation of prompt2prompt
        # def step_callback(self, x_t):
        #     self.cur_att_layer = 0
        #     self.cur_step += 1
        #     self.between_steps() # NOTE between_steps is called in step_callback, which differs from the implementation here
        #     return x_t


    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        """[Copied]I guess the diffusion of google has some unconditional attention layer
        No unconditional attention layer in Stable diffusion
        """
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # this function will be called in P2PCrossAttnProcessor.__call__
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet) # classifier-free guidance：因为此时的attn_map中即包含了text_prompt的，也包含了null text prompt的。但是我们存储的目标是text_prompt的attn_map（因为空文本的corss_attn map不能反映语义layout，没有什么用）。因此，我们对attn_map的形状进行划分，取attn_map的后半部分，即text_prompt的attn_map，调用forward方法进行处理。
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1 # assigned in `register_attention_control` function
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # will not change attn, just store it
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        """divide the attention map value in attention store by denoising steps"""
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

class LocalBlend:
    def __call__(self, x_t, attention_store): # x_t: torch.Size([2, 4, 64, 64]).        x_t.shape[2:] == torch.Size([64, 64])
        k = 1 # self.alpha_layers: torch.Size([2, 1, 1, 1, 1, 77])
        batch_size = x_t.shape[0]
        assert batch_size == 1, "batch_size > 1 is not supported yet."
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3] # List[torch.Size([batchsize 2, 8, 256, 77])]. 5 maps in total, so 5*8=40 below.
        maps = [item.reshape(batch_size, -1, 1, 16, 16, self.max_num_words) for item in maps] # 16*16=256
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1) # (maps * self.alpha_layers).sum(-1): torch.Size([2, 40, 1, 16, 16]).     (maps * self.alpha_layers).sum(-1).mean(1): torch.Size([2, 1, 16, 16])
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = F.interpolate(mask, size=(x_t.shape[2:])) # torch.Size([2, 1, 64, 64])
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        
        x_t = x_t * mask
        return x_t

    def __init__(
        self,
        prompt: str,
        words: List[str],
        tokenizer,
        device,
        threshold=0.3,
        max_num_words=77,
    ):
        self.max_num_words = max_num_words

        alpha_layers = torch.zeros(1, 1, 1, 1, 1, self.max_num_words)
        for word in words:
            ind = get_word_inds(prompt, word, tokenizer)
            alpha_layers[:, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class LocalBlendWithNegativeWords:
    def __call__(self, x_t, attention_store): # x_t: torch.Size([2, 4 what?, 64, 64]).        x_t.shape[2:] == torch.Size([64, 64])
        k = 1 # self.alpha_layers: torch.Size([2, 1, 1, 1, 1, 77])
        batch_size = x_t.shape[0]
        assert batch_size == 1, "batch_size > 1 is not supported yet."
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3] # List[torch.Size([batchsize 2, 8, 256, 77])]. 5 maps in total, so 5*8=40 below.
        maps = [item.reshape(batch_size, -1, 1, 16, 16, self.max_num_words) for item in maps] # 16*16=256
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers["positive"]).sum(-1) - (maps * self.alpha_layers["negative"]).sum(-1)
        
        maps = maps.mean(1)
        
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = F.interpolate(mask, size=(x_t.shape[2:])) # torch.Size([2, 1, 64, 64])
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0] # 归一化
        mask = mask.gt(self.threshold)
        
        
        x_t = x_t * mask
        
        eval_utils = {}
        eval_utils["mask"] = mask
        return x_t, eval_utils

    def __init__(
        self,
        prompt: str,
        words: List[str],
        neg_words: List[str],
        tokenizer,
        device,
        threshold=0.3,
        max_num_words=77,
    ):
        self.max_num_words = max_num_words
        self.alpha_layers = {}
        alpha_layers = torch.zeros(1, 1, 1, 1, 1, self.max_num_words)
        for word in words:
            ind = get_word_inds(prompt, word, tokenizer)
            assert len(ind) > 0, f"Cannot find {word} in {prompt}"
            alpha_layers[:, :, :, :, :, ind] = 1
        self.alpha_layers["positive"] = alpha_layers.to(device)
        
        alpha_layers = torch.zeros(1, 1, 1, 1, 1, self.max_num_words)
        if neg_words is not None and len(neg_words) > 0:
            for word in neg_words:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[:, :, :, :, :, ind] = 1
        self.alpha_layers["negative"] = alpha_layers.to(device)
        
        self.threshold = threshold


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if isinstance(word_place, str):
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif isinstance(word_place, int):
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)
