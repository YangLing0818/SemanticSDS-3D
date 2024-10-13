import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gs.gaussian_splatting import GaussianSplattingRenderer
from utils.spiral import get_c2w_from_up_and_look_at
from utils.camera import CameraInfo
from utils.colormaps import apply_depth_colormap


def take_photo_from_ckpt(ckpt, save_dir="./user_guided", save_name=None):
    ckpt = torch.load(ckpt, map_location="cpu")
    cfg = ckpt["cfg"]
    prompt = cfg["prompt"]["prompt"]

    renderer = GaussianSplattingRenderer.load(None, ckpt).to("cuda")

    renderer.eval()

    camera_info = CameraInfo.from_reso(1024)

    up = np.array([0.0, 0.0, 1.0])
    look_at = np.array([0.0, 0.0, 0.0])

    # front = np.array([1.5, -1.5, 2.0]) * 1.1
    left = np.array([1.5, 1, 2.0])
    right = np.array([-1.5, -1, 2.0])

    front = np.array([2.5, 0.0, 1.5])

    front = get_c2w_from_up_and_look_at(up, look_at, front, return_pt=True).to("cuda")
    left = get_c2w_from_up_and_look_at(up, look_at, left, return_pt=True).to("cuda")
    right = get_c2w_from_up_and_look_at(up, look_at, right, return_pt=True).to("cuda")

    poses = {
        "front": front,
        "left": left,
        "right": right,
    }

    prompt = prompt.replace(" ", "_")

    if save_name is None:
        save_dir = Path(save_dir) / prompt
    else:
        save_dir = Path(save_dir) / save_name
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    for name, pose in poses.items():
        with torch.no_grad():
            out = renderer.render_one(pose, camera_info, return_T=True)
            rgb = (out["rgb"].cpu().numpy() * 255).astype(np.uint8)
            depth = (
                apply_depth_colormap(out["depth"], out["opacity"]).cpu().numpy() * 255.0
            ).astype(np.uint8)
            T = 255 - (out["T"].cpu().numpy() * 255).astype(np.uint8)[..., 0]

            image = Image.fromarray(rgb)
            image = image.convert("RGBA")
            image.putalpha(Image.fromarray(T))

            image.save(save_dir / f"{name}_rgb.png")

            depth = Image.fromarray(depth)
            depth = depth.convert("RGBA")
            depth.putalpha(Image.fromarray(T))

            depth.save(save_dir / f"{name}_depth.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--save_dir", type=str, default="./guided")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--step", type=str, default=None)

    opt = parser.parse_args()

    ckpt = Path(opt.ckpt)
    if not ckpt.exists():
        uid, time, day, prompt = str(ckpt).strip().split("|")

        ckpt_dir = Path(f"./checkpoints/{prompt}/{day}/{time}/ckpts/")
        if opt.step is None:
            files = ckpt_dir.glob("*.pt")
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            ckpt = latest_file
        else:
            ckpt = ckpt_dir / f"step_{opt.step}.pt"

    take_photo_from_ckpt(ckpt, opt.save_dir, opt.save_name)
