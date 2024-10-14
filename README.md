# Semantic Score Distillation Sampling for Compositional Text-to-3D Generation
<p align="left">
  <a href='https://arxiv.org/abs/2410.09009'>
  <img src='https://img.shields.io/badge/Arxiv-2410.09009-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  </p>
  
**TL;DR:** Official repo of [SemanticSDS](https://arxiv.org/abs/2410.09009). By leveraging program-aided layout planning, augmenting 3D Gaussians with semantic embeddings, and guiding SDS with rendered semantic maps, SemanticSDS unlocks the compositional capabilities of pre-trained diffusion models, generating complex 3D scenes comprising multiple objects with various attributes.
<details>
    <summary>Click for full abstract</summary>
Generating high-quality 3D assets from textual descriptions remains a pivotal challenge in computer graphics and vision research. Due to the scarcity of 3D data, state-of-the-art approaches utilize pre-trained 2D diffusion priors, optimized through Score Distillation Sampling (SDS). Despite progress, crafting complex 3D scenes featuring multiple objects or intricate interactions is still difficult. To tackle this, recent methods have incorporated box or layout guidance. However, these layout-guided compositional methods often struggle to provide fine-grained control, as they are generally coarse and lack expressiveness. To overcome these challenges, we introduce a novel SDS approach, Semantic Score Distillation Sampling (SemanticSDS), designed to effectively improve the expressiveness and accuracy of compositional text-to-3D generation. Our approach integrates new semantic embeddings that maintain consistency across different rendering views and clearly differentiate between various objects and parts. These embeddings are transformed into a semantic map, which directs a region-specific SDS process, enabling precise optimization and compositional generation. By leveraging explicit semantic guidance, our method unlocks the compositional capabilities of existing pre-trained diffusion models, thereby achieving superior quality in 3D content generation, particularly for complex objects and scenes. Experimental results demonstrate that our SemanticSDS framework is highly effective for generating state-of-the-art complex 3D content.
</details>

## Video results

https://github.com/user-attachments/assets/c321b2b8-ad6e-4eb1-bf6d-00445bb4348d

A rabbit sits atop a large, expensive watch with many shiny gears, made half of iron and half of gold, eating a birthday cake that is in front of the rabbit.

https://github.com/user-attachments/assets/803d6cea-2f30-42b6-a3ca-aff2eb256d64

A corgi is positioned to the left of a LEGO house, while a car with its front half made of cheese and its rear half made of sushi is situated to the right of the house made of LEGO.

https://github.com/user-attachments/assets/8fa7ff7c-3f3c-47c2-8175-2f3c4563d549

A hamburger, a loaf of bread, an order of fries, and a cup of Coke.

https://github.com/user-attachments/assets/ad6c607a-fb8c-4d52-848c-cc6dfa19d0df

A cozy scene with a plush triceratops toy surrounded by a plate of chocolate chip cookies, a glistening cinnamon roll, and a flaky croissant.

<details>
    <summary>More results</summary>

https://github.com/user-attachments/assets/9688d8e1-6ac7-49e6-abfb-ca649a1b32c1

A mannequin adorned with a dress made of feathers and moss stands at the center, flanked by a vase with a single blue tulip and another with blue roses.

https://github.com/user-attachments/assets/6fdbce53-5146-4b23-adfe-2cfb66208c1b

A pyramid-shaped burrito artistically blended with the Great Pyramid.

https://github.com/user-attachments/assets/2e0d2aed-c771-4783-b5ca-5406e1187859

A train with a front made of cake and a back of a steam engine.
</details>

## Quick Start

1. **Install the requirements:**

    It is recommended to use CUDA 11.8, but other versions should also work fine.
    ```bash
    conda create -n semanticSDS python=3.9
    conda activate semanticSDS
    pip install -r requirements.txt
    ```

    To install PyTorch3D, follow the instructions from [PyTorch3D's official installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). Here is a quick rundown to install it from a stable branch with CUDA support:

    ```bash
    git clone --branch stable https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    FORCE_CUDA=1 pip install .
    ```

   (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:

    ```sh
    pip install ninja
    ```

2. **Build the extension for Gaussian Splatting:**

    ```bash
    cd gs
    ./build.sh
    ```

3. **Start!**
    Run the main program using the following command. Make sure to specify the appropriate config name:
    ```bash
    python main.py --config-name=pyramid
    ```

## Program-aided layout planning

Add your OpenAI API key in the generate_layouts_PAL.py file at the location marked with a `# openai token` comment.

### Single Prompt Usage

To generate a layout for a single user prompt, use the following command:

```bash
python generate_layouts_PAL.py --template_version "v0.14_PAL_updated_incontext" --llm_name "gpt-4-32k" --user_prompt "A corgi is situated to the left of a house, while a car is positioned to the right of the house. The car above is split into two layers along the depth axis. The front layer of the car is constructed from wood. The left half of the rear layer is made of sushi, and the right half is made of cheese."
```

You can view the results in the layouts_cache folder.

### Batch Prompt Usage

For processing multiple prompts, you can use a batch file containing the prompts. Use the following command, replacing `./prompts_to_ask.txt` with the path to your text file:

```bash
python generate_layouts_PAL.py --template_version "v0.14_PAL_updated_incontext" --llm_name "gpt-4-32k" --batch_prompt_file "./prompts_to_ask.txt"
```

If you are not satisfied with the responses obtained before, add `--disable_response_cache` to your command.

## Running the Main Program

1. **Configuration**: Edit the `.yaml` files in the `conf` folder:
    - Set `llm_init.enabled` to `true`.
    - Update `json_path` to point to the `normalized.json` file for the corresponding layouts:

    ```
    llm_init:
      enabled: true
      json_path: layouts_cache/v0.14_PAL_updated_incontext/gpt-4-32k/mannequin/normalized.json
    ```

2. **Execute**: Run the main program with the appropriate config name:

    ```bash
    python main.py --config-name=mannequin
    ```

## Tips

- To resolve `ImportError: libXrender.so.1: cannot open shared object file: No such file or directory`, try:

    ```bash
    apt-get update && apt-get install libxrender1
    ```

- To resume from a checkpoint:

    - Add `+ckpt=<path_to_your_ckpt>` to the run command, where `<path_to_your_ckpt>` is the actual path to your `.pt` checkpoint file.
    - Ensure that the `.yaml` configuration file you're using is the same as the one used to save the checkpoint.

    Example command to resume from a checkpoint:

    ```bash
    python main.py --config-name=car +ckpt="checkpoints/a_dslr_photo_of_a_car_made_out_of_lego/2024-10-13/080848/ckpts/step_2000.pt"
    ```

- Enable wandb: To monitor and log your runs using [Weights & Biases (wandb.ai)](https://wandb.ai/site), add `wandb=true` to the run command.

## Acknowledgement

Thanks to the awesome open-source projects of [GSGEN](https://github.com/gsgen3d/gsgen) for their outstanding work, which have significantly contributed to this codebase.

## Citation
```
@article{yang2024semanticsds,
  title={Semantic Score Distillation Sampling for Compositional Text-to-3D Generation},
  author={Yang, Ling and Zhang, Zixiang and Han, Junlin and Zeng, Bohan and Li, Runjia and Torr, Philip and Zhang, Wentao},
  journal={arXiv preprint arXiv:2410.09009},
  year={2024}
}
```

