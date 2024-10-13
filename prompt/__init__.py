from .stable_diffusion_prompt import StableDiffusionPromptProcessor
from .deep_floyd_prompt import DeepFloydPromptProcessor
from .focused_stable_diffusion_prompt import FocusedStableDiffusionPromptProcessor

prompt_processors = dict(
    stable_diffusion=StableDiffusionPromptProcessor,
    deep_floyd=DeepFloydPromptProcessor,
    focused_stable_diffusion=FocusedStableDiffusionPromptProcessor,
)


def get_prompt_processor(cfg, **kwargs):
    try:
        return prompt_processors[cfg.type](cfg, **kwargs)
    except KeyError:
        raise NotImplementedError(f"Prompt processor {cfg.type} not implemented.")
