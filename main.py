import os
import hydra
from trainer import Trainer
from omegaconf import OmegaConf
from rich.console import Console
import json

console = Console()


@hydra.main(version_base="1.3", config_path="conf", config_name="trainer")
def main(cfg):

    if 'llm_init' in cfg and cfg.llm_init.enabled:
        with open(cfg.llm_init.json_path, 'r') as file:
            llm_init_json_obj = json.load(file)
        # overwrite subinit
        tmp_subinit = OmegaConf.create(cfg.init.subinit[0])
        cfg.init.subinit.clear()
        for entity in llm_init_json_obj["scene_models"]:
            new_subinit = OmegaConf.create(tmp_subinit)
            new_subinit.prompt = entity["stable diffusion prompt"]
            new_subinit.xyz_offset = [entity["position"]["x"], entity["position"]["y"], entity["position"]["z"]]
            new_subinit.bbox_dimensions = [entity["dimensions"]["x"], entity["dimensions"]["y"], entity["dimensions"]["z"]]
            new_subinit.part_space_ratios = []
            new_subinit.part_specific_guidance_prompts = []
            for part in entity["parts_in_local_01_coords"]:
                new_subinit.part_space_ratios.append([part["min_corner"], part["max_corner"]])
                new_prompt = OmegaConf.create(tmp_subinit.part_specific_guidance_prompts[0])
                new_prompt.prompt = part["prompt"]
                new_subinit.part_specific_guidance_prompts.append(new_prompt)
            cfg.init.subinit.append(new_subinit)
        cfg.num_groups = -1 # for automatically setting num_groups in init.py
        console.print(
            '[red]llm_init is enabled, num_groups is set to -1, subinit is overwritten by llm_init.json[/red]'
        )
        
        # save overrided config
        json_dir = os.path.dirname(cfg.llm_init.json_path)
        config_save_path = os.path.join(json_dir, "overrided_config.yaml")
        
        with open(config_save_path, "w") as f:
            OmegaConf.save(config=cfg, f=f)
        
        console.print(f"[green]Overrided configuration saved to: {config_save_path}[/green]")
        
    cfg.prompt = cfg.init.subinit[0].part_specific_guidance_prompts[0]
     
    ###########################    
    ### automatically set num_groups ###
    num_groups = 0
    for subinit in cfg.init.subinit:
        assert len(subinit.part_space_ratios) == len(subinit.part_specific_guidance_prompts)
        num_groups += len(subinit.part_space_ratios)
    if cfg.num_groups <= 0:
        cfg.num_groups = num_groups
        console.print(
            '[red]num_groups is overwrited as {}[/red]'.format(cfg.num_groups)
        )
    else:
        assert cfg.num_groups == num_groups, 'num_groups is not equal to the sum of part_space_ratios in subinit'
        
        
    upsample_tune_only: bool = cfg.get("upsample_tune_only", False)
    # console.print(OmegaConf.to_yaml(cfg, resolve=True))
    ckpt = cfg.get("ckpt", None)
    if not upsample_tune_only:
        if ckpt is not None:
            trainer = Trainer.load(cfg.ckpt, cfg)
        else:
            trainer = Trainer(cfg)
        trainer.train_loop()

        if hasattr(cfg, "upsample_tune") and cfg.upsample_tune.enabled == True:
            trainer.tune_with_upsample_model()
    else:
        assert (
            ckpt is not None
        ), "ckpt must be specified when upsample_tune_only is True"
        console.print("[red]Tune from ckpt: {}[/red]".format(ckpt))
        trainer = Trainer.load(cfg.ckpt, cfg)

        trainer.tune_with_upsample_model()

        return 0


if __name__ == "__main__":
    main()
