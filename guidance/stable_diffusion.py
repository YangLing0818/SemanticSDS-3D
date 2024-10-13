from . import BaseGuidance
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionPipeline,
    PNDMScheduler,
)

from utils.typing import *
from utils.ops import perpendicular_component
from utils.misc import C
from utils.prompt2prompt import (
    P2PCrossAttnProcessor,
    AttentionStore,
)
from rich.console import Console

console = Console()

# from prompt.prompt_processors import BasePromptProcessor, PromptEmbedding


# class StableDiffusionPromptProcessor(BasePromptProcessor):
#     def prepare_text_encoder(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.pretrained_model_name_or_path, subfolder="tokenizer"
#         )
#         self.text_encoder = CLIPTextModel.from_pretrained(
#             self.pretrained_model_name_or_path,
#             subfolder="text_encoder",
#             device_map="auto",
#         )

#     def encode_prompts(self, prompts):
#         with torch.no_grad():
#             print(prompts)
#             tokens = self.tokenizer(
#                 prompts,
#                 padding="max_length",
#                 max_length=self.tokenizer.model_max_length,
#                 return_tensors="pt",
#             ).to(self.device)
#             # print(tokens.input_ids.device)
#             text_embeddings = self.text_encoder(tokens.input_ids)[0]

#         return text_embeddings


class StableDiffusionGuidance(BaseGuidance):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        if self.cfg.keep_complete_pipeline:
            pipe_kwargs = {
                "torch_dtype": self.weights_dtype,
            }
        else:
            pipe_kwargs = {
                "tokenizer": None,
                "safety_checker": None,
                "feature_extractor": None,
                "requires_safety_checker": False,
                "torch_dtype": self.weights_dtype,
                "cache_dir": "./.cache",
            }

        if self.cfg.repeat_until_success:
            success = False
            while not success:
                try:
                    self.pipe = StableDiffusionPipeline.from_pretrained(
                        self.cfg.pretrained_model_name_or_path,
                        **pipe_kwargs,
                    ).to(self.device)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    console.print(".", end="")
                else:
                    success = True
                    break
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                **pipe_kwargs,
            ).to(self.device)

        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        
        # # replace self.unet's CrossAttnProcessor with P2PCrossAttnProcessor
        # # so that attention map will be stored in self.attention_store when self.forward_unet
        # self.attention_store = AttentionStore()
        # self.register_attention_control(self.attention_store)


        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        # TODO: make this configurable
        scheduler = self.cfg.scheduler.type.lower()
        if scheduler == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )
        elif scheduler == "pndm":
            self.scheduler = PNDMScheduler(**self.cfg.scheduler.args)
        else:
            raise NotImplementedError(f"Scheduler {scheduler} not implemented")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.step = 0
        self.max_steps = self.cfg.max_steps
        self.set_min_max_steps()
        self.grad_clip_val = None
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        if self.cfg.enable_attention_slicing:
            # enable GPU VRAM saving, reference: https://huggingface.co/stabilityai/stable-diffusion-2
            self.pipe.enable_attention_slicing(1)

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self):
        min_step_percent = C(self.cfg.min_step_percent, self.step, self.max_steps)
        max_step_percent = C(self.cfg.max_step_percent, self.step, self.max_steps)
        self.min_t_step = int(self.num_train_timesteps * min_step_percent)
        self.max_t_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents,
        t,
        encoder_hidden_states,
    ):
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(self, imgs):
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents,
        latent_height: int = 64,
        latent_width: int = 64,
    ):
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        latents, # torch.Size([batchsize, channel 4, resolution 64, resolution 64])
        t,
        prompt_embedding,
        elevation,
        azimuth,
        camera_distances,
        noise=None,
        object_center=None,
        camera_position=None,
        dict4obj_specific_view=None,
        obj2cam_under_local=None,
    ):
        batch_size = elevation.shape[0]

        if prompt_embedding.use_perp_negative:
            raise NotImplementedError("for mask from attention, use_perp_negative is not supported yet.")
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_embedding.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.use_view_dependent_prompt
            )
            with torch.no_grad():
                if noise is None:
                    noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
            noise_pred = noise_pred.to(torch.float32)
        else:
            neg_guidance_weights = None
            if self.cfg.get("use_object_dependent_view", False) and dict4obj_specific_view["is_goto_global"]:                
                text_embeddings = prompt_embedding.get_text_embedding(
                    elevation, azimuth, camera_distances, self.cfg.use_view_dependent_prompt, 
                    object_center=object_center, 
                    camera_position=camera_position,
                    use_object_dependent_view=self.cfg.use_object_dependent_view,
                    obj2cam_under_local=obj2cam_under_local,
                )
            else:
                text_embeddings = prompt_embedding.get_text_embedding(
                    elevation, azimuth, camera_distances, self.cfg.use_view_dependent_prompt
                )
            
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise.    noise: torch.Size([batchsize, channel 4, resolution 64, resolution 64])
                if noise is None:
                    noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0) # why need "* 2"? Maybe, one for predicting noise_pred_text, and the other for predicting noise_pred_uncond
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings, # encoder_hidden_states seems to be for `k` and `v` in attention mechanism
                ) # torch.Size([batchsize * 2, channel 4, 64, 64])

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2) # noise_pred_text: torch.Size([batchsize, channel 4, 64, 64]), noise_pred_uncond: torch.Size([batchsize, channel 4, 64, 64])
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            ) # torch.Size([batchsize, channel 4, 64, 64])

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1) # self.alphas[t]: torch.Size([4]). -> w: torch.Size([4, 1, 1, 1])
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )
            
        
        
        # noise_pred: torch.Size([batch_size? 4, 4, 64, 64]), noise: torch.Size([4, 4, 64, 64])
        grad = w * (noise_pred - noise)
        # grad: torch.Size([4, 4, 64, 64])
        
        
        #### mask grad with attention map #### NOTE NOT used ####
        # Average on unet foward steps. But since we only have one step, 
        # it's the same as the attention map stored
        local_blends_eval_utils = []
        if hasattr(prompt_embedding, "get_local_blend"):
            attention_maps = self.attention_store.get_average_attention()
            attention_maps = {
                key: [item.reshape(batch_size, -1, *item.shape[1:]) for item in attention_maps[key]] for key in attention_maps
            }
            
            assert batch_size == attention_maps["down_cross"][0].shape[0]
            attention_maps_by_batch = [{} for _ in range(batch_size)]
            for place, attention_maps_on_the_place in attention_maps.items():
                for batch_index in range(batch_size):
                    attention_maps_by_batch[batch_index][place] = [attention_map[batch_index].unsqueeze(0) for attention_map in attention_maps_on_the_place] # .unsqueeze(0) for keep dim, adjusting for localblend.__call__
            
            
            local_blends = prompt_embedding.get_local_blend(
                elevation, azimuth, camera_distances, self.cfg.use_view_dependent_prompt
            )
            assert batch_size == len(local_blends)
            assert batch_size == grad.shape[0]
            for i in range(batch_size):
                grad[i:i+1], lb_eval_utils = local_blends[i](grad[i].unsqueeze(0), attention_maps_by_batch[i])
                local_blends_eval_utils.append(lb_eval_utils)

        
        
        guidance_eval_utils = {
            "use_perp_neg": prompt_embedding.use_perp_negative,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "local_blends": local_blends_eval_utils,
        }

        return grad, guidance_eval_utils

    def forward(
        self,
        rgb,
        prompt_embedding_list,
        elevation,
        azimuth,
        camera_distance,
        rgb_as_latents=False,
        guidance_eval=False,
        prompt_mask_list=None, # prompt_mask_list: a list of torch.Size([bs 4, H 512, W 512])?
        global_object_center_list=None,
        camera_position=None,
        dict4obj_specific_view=None,
        **kwargs,
    ):
        assert len(prompt_embedding_list) == len(prompt_mask_list), f"{len(prompt_embedding_list)} != {len(prompt_mask_list)}. Might be caused by the prompt_scheduler returning num_groups and other ids together"
        assert len(prompt_embedding_list) == len(global_object_center_list), f"{len(prompt_embedding_list)} != {len(global_object_center_list)}."
        bs = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2) # -> rgb_BCHW: torch.Size([batch_size, 3, 512, 512])
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            ) # torch.Size([3, 3, 512, 512])
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512) # -> latents: torch.Size([batch_size, 4, 64, 64])

        t = torch.randint(
            self.min_t_step,
            self.max_t_step + 1,
            [bs],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        compositional_grad = [] # total_grad@torch.float32
        foreground_mask = torch.zeros_like(latents, dtype=torch.bool) 
        for id in range(len(prompt_embedding_list)):
            prompt_embedding = prompt_embedding_list[id]
            object_center = global_object_center_list[id]
            obj2cam_under_local = dict4obj_specific_view["batched_obj2cam_under_local"][id]
            grad, guidance_eval_utils = self.compute_grad_sds(
                latents, t, prompt_embedding, elevation, azimuth, camera_distance,
                noise = noise,
                object_center = object_center,
                camera_position=camera_position,
                dict4obj_specific_view=dict4obj_specific_view,
                obj2cam_under_local=obj2cam_under_local,
            ) # grad@torch.Size([bs 4, c 4, h 64, w 64])
            
            prompt_mask = prompt_mask_list[id] # prompt_mask@torch.Size([bs, 64, 64]) torch.bool
            prompt_mask = prompt_mask.unsqueeze(1).expand(-1, 4, -1, -1) # ->@torch.Size([bs 4, 4, h 64, w 64])
            
            # compositional_grad = compositional_grad + grad * prompt_mask # torch.Size([bs 4, c 4, h 64, w 64])
            compositional_grad.append(grad * prompt_mask)
            
            foreground_mask = torch.logical_or(foreground_mask, prompt_mask.bool())
        
        # .mean(dim=0) for averaging the grad from different prompts
        compositional_grad = torch.stack(compositional_grad, dim=0).mean(dim=0) # ->@torch.Size([4, 4, 64, 64])?
        compositional_grad = torch.nan_to_num(compositional_grad)
        # latents = latents * foreground_mask.float()
        # total_grad = total_grad * foreground_mask.float()
        
        if self.grad_clip_val is not None:
            compositional_grad = compositional_grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        # latents.require_grad: True    grad.requires_grad: False
        target = (latents - compositional_grad).detach() # latents: torch.Size([4, 4, 64, 64]), grad: torch.Size([4, 4, 64, 64]) -> target: torch.Size([4, 4, 64, 64])
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / bs # = 0.5 * 2 * grad^2 / bs?
        loss_sds_each = 0.5 * F.mse_loss(latents, target, reduction="none").sum(
            dim=[1, 2, 3]
        ) # loss_sds on each batch, torch.Size([4])?

        guidance_out = {
            "loss_sds": loss_sds,
            "loss_sds_each": loss_sds_each,
            "grad_norm": compositional_grad.norm(),
            "min_t_step": self.min_t_step,
            "max_t_step": self.max_t_step,
        }

        if guidance_eval:
            # guidance_eval_out = self.guidance_eval(**guidance_eval_utils) # https://github.com/threestudio-project/threestudio/blob/cd462fb0b73a89b6be17160f7802925fe6cf34cd/threestudio/models/guidance/stable_diffusion_guidance.py#L535
            # texts = []
            # for n, e, a, c in zip(
            #     guidance_eval_out["noise_levels"], elevation, azimuth, camera_distance
            # ):
            #     texts.append(
            #         f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
            #     )
            # guidance_eval_out.update({"texts": texts})
            # guidance_out.update({"eval": guidance_eval_out})
            guidance_out.update({"guidance_eval_utils": guidance_eval_utils}) # FIXME should change the value from guidance_eval_utils to guidance_eval_utils_list, and should modify outside the function too

        return guidance_out

    # def step(self, epoch: int, step: int):
    #     if self.cfg.grad_clip is not None:
    #         self.grad_clip_val = C(self.cfg.grad_clip, epoch, step)

    # vanilla scheduler use constant min max steps
    # self.set_min_max_steps()

    def update(self, step):
        self.step = step
        self.set_min_max_steps()
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, step, self.max_steps)

    def log(self, writer, step):
        writer.add_scalar("guidance/min_step", self.min_t_step, step)
        writer.add_scalar("guidance/max_step", self.max_t_step, step)

    def register_attention_control(self, controller):
        """
        Modified from https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_prompt2prompt.py
        just change self.unet through self.unet.set_attn_processor
        """
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            (None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim) # https://github.com/bloc97/CrossAttentionControl/issues/26
            if name.startswith("mid_block"):
                self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                list(reversed(self.unet.config.block_out_channels))[block_id]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(controller=controller, place_in_unet=place_in_unet)

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count