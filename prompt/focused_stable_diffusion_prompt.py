from .prompt_processors import PromptEmbedding
from .stable_diffusion_prompt import StableDiffusionPromptProcessor
from utils.prompt2prompt import get_word_inds, LocalBlend, LocalBlendWithNegativeWords
from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class FocusedPromptEmbedding(PromptEmbedding):
    local_blend_view_dependent_list: List[LocalBlendWithNegativeWords] = field(default_factory=list) # bug fixed: non-default argument 'local_blend_view_dependent_list' follows default argument
    
    def get_local_blend(
        self,
        elevation, # torch.Size([batch_size])
        azimuth, # torch.Size([batch_size])
        camera_distances, # torch.Size([batch_size])
        use_view_dependent_prompt=False,
    ):

        if use_view_dependent_prompt:
            direction_idx = torch.zeros_like(elevation, dtype=torch.long) # -> torch.Size([batch_size])
            for d in self.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances) # d.condition returns a boolean torch.Size([5])
                ] = self.direction2idx[d.name]

            local_blend = [self.local_blend_view_dependent_list[int(i)] for i in direction_idx]
        else:
            raise NotImplementedError

        return local_blend


class FocusedStableDiffusionPromptProcessor(StableDiffusionPromptProcessor):
    def __init__(self, cfg, guidance_model=None):
        super().__init__(cfg, guidance_model)
        
        # self.prompts_view_dependent is List[str], length is 4
        local_blend_view_dependent_list = []
        for i in range(len(self.prompts_view_dependent)):
            prompt_for_a_view = self.prompts_view_dependent[i]
            local_blend_view_dependent_list.append(LocalBlendWithNegativeWords(
                prompt=prompt_for_a_view,
                words=cfg.attention_mask.focus_words,
                neg_words= cfg.attention_mask.get("negative_focus_words", []),
                tokenizer=self.tokenizer,
                device=self.device,
                threshold=cfg.attention_mask.activation_threshold,
                max_num_words=cfg.attention_mask.max_num_words,
            ))
        self.local_blend_view_dependent_list = local_blend_view_dependent_list
            
    def get_prompt_embedding(self) -> FocusedPromptEmbedding:
        return FocusedPromptEmbedding(
            text_embedding=self.text_embedding,
            uncond_text_embedding=self.uncond_text_embedding,
            text_embedding_view_dependent=self.text_embedding_view_dependent,
            uncond_text_embedding_view_dependent=self.uncond_text_embedding_view_dependent,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_negative=self.cfg.use_perp_negative,
            debug=self.cfg.debug,
            local_blend_view_dependent_list=self.local_blend_view_dependent_list
        )