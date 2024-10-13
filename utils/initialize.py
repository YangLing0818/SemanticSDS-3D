from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.misc import print_info
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor
from utils.ops import lift_to_3d, farthest_point_sampling
from utils.camera import CameraInfo
from utils.mesh import load_mesh_as_pcd, load_mesh_as_pcd_trimesh
from rich.console import Console

console = Console()


def nearest_neighbor_initialize(pts, k=3):
    import faiss

    ## set cov to mean distance of nearest k points
    if not isinstance(pts, torch.Tensor):
        pts = torch.from_numpy(pts).to("cuda")

    # pts = pts.to("cuda")
    # dist = torch.cdist(pts, pts)
    # topk = torch.topk(dist, k=k, dim=1, largest=False)

    # return topk.mean(axis=)
    res = faiss.StandardGpuResources()
    # pts = pts.to("cuda")
    index = faiss.IndexFlatL2(pts.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index_flat.add(pts.cpu())
    D, _ = gpu_index_flat.search(pts, k + 1)

    return torch.from_numpy(D[..., 1:].mean(axis=1))


def get_qvec(cfg):
    qvec = torch.zeros(cfg.num_points, 4, dtype=torch.float32)
    qvec[:, 0] = 1.0
    return qvec


def get_svec(cfg):
    svec = torch.ones(cfg.num_points, 3, dtype=torch.float32) * cfg.svec_val
    return svec


def get_alpha(cfg):
    alpha = torch.ones(cfg.num_points, dtype=torch.float32) * cfg.alpha_val
    return alpha


def base_initialize(cfg):
    initial_values = {}
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["svec"] = get_svec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    initial_values["color"] = torch.rand(cfg.num_points, 3, dtype=torch.float32)
    initial_values["mean"] = (
        torch.randn(cfg.num_points, 3, dtype=torch.float32) * cfg.mean_std
    )

    return initial_values


def unisphere_initialize(cfg):
    R = cfg.mean_std
    N = cfg.num_points
    theta = torch.rand(N) * 2 * np.pi
    phi = torch.rand(N)
    phi = torch.acos(1 - 2 * phi) # why not phi = torch.rand(N) * np.pi
    x = R * torch.sin(phi) * torch.cos(theta)
    y = R * torch.sin(phi) * torch.sin(theta)
    z = R * torch.cos(phi)

    initial_values = {}
    initial_values["mean"] = torch.stack([x, y, z], dim=1)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["svec"] = get_svec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    initial_values["color"] = torch.rand(cfg.num_points, 3, dtype=torch.float32)

    return initial_values


def semisphere_initialize(cfg):
    R = cfg.mean_std
    N = cfg.num_points
    theta = torch.rand(N) * np.pi + np.pi / 2.0
    phi = torch.rand(N)
    phi = torch.acos(1 - 2 * phi)
    x = R * torch.sin(phi) * torch.cos(theta)
    y = R * torch.sin(phi) * torch.sin(theta)
    z = R * torch.cos(phi)

    initial_values = {}
    initial_values["mean"] = torch.stack([x, y, z], dim=1)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["svec"] = get_svec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    initial_values["color"] = torch.rand(cfg.num_points, 3, dtype=torch.float32)

    return initial_values

def point_e_initialize(cfg):
    if not hasattr(point_e_initialize, 'initialized'):
        from utils.point_e_helper import point_e_generate_pcd_from_text
        point_e_initialize.initialized = True
        point_e_initialize.generate_pcd = point_e_generate_pcd_from_text

    prompt = cfg.prompt
    pcd = point_e_initialize.generate_pcd(prompt)
    xyz, rgb = pcd[:, :3], pcd[:, 3:]

    if cfg.num_points > 4096:
        if cfg.get("random_exceed", False):
            indices = torch.randint(
                0, xyz.size(0), (cfg.num_points,), device=xyz.device
            )
            xyz = xyz[indices]
            rgb = rgb[indices]
        else:
            extra_xyz = (
                torch.randn(
                    cfg.num_points - 4096, 3, dtype=torch.float32, device=xyz.device
                )
                * cfg.mean_std
            )
            extra_rgb = torch.rand(
                cfg.num_points - 4096, 3, dtype=torch.float32, device=rgb.device
            )
            xyz = torch.cat([xyz, extra_xyz], dim=0)
            rgb = torch.cat([rgb, extra_rgb], dim=0)

    xyz -= xyz.mean(dim=0, keepdim=True)

    xyz = xyz / (xyz.norm(dim=-1).max() + 1e-5)
    xyz = xyz * cfg.mean_std

    if cfg.get("facex", False):
        # align the point cloud to the x axis
        console.print("[red]will align the point cloud to the x axis")
        x, y, z = xyz.chunk(3, dim=-1)
        xyz = torch.cat([-y, x, z], dim=-1)

    if cfg.get("random_color", False):
        console.print("[red]will use random color")
        rgb = torch.rand_like(rgb)

    if cfg.get("white_color", False):
        console.print("[red]will make all the gaussians white, for experimental usage")
        rgb = torch.ones_like(rgb) * 0.7

    z_scale = cfg.get("z_scale", 1.0)
    xyz[..., 2] *= z_scale

    initial_values = {}
    initial_values["mean"] = xyz
    initial_values["color"] = rgb
    # breakpoint()
    initial_values["svec"] = get_svec(cfg)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    return initial_values

def point_e_initialize_with_bbox(cfg):
    if not hasattr(point_e_initialize_with_bbox, 'initialized'):
        from utils.point_e_helper import point_e_generate_pcd_from_text
        point_e_initialize_with_bbox.initialized = True
        point_e_initialize_with_bbox.generate_pcd = point_e_generate_pcd_from_text

    prompt = cfg.prompt
    pcd = point_e_initialize_with_bbox.generate_pcd(prompt)
    xyz, rgb = pcd[:, :3], pcd[:, 3:]

    if cfg.num_points > 4096:
        if cfg.get("random_exceed", False):
            indices = torch.randint(
                0, xyz.size(0), (cfg.num_points,), device=xyz.device
            )
            xyz = xyz[indices]
            rgb = rgb[indices]
        else:
            extra_xyz = (
                torch.randn(
                    cfg.num_points - 4096, 3, dtype=torch.float32, device=xyz.device
                )
                * cfg.mean_std
            )
            extra_rgb = torch.rand(
                cfg.num_points - 4096, 3, dtype=torch.float32, device=rgb.device
            )
            xyz = torch.cat([xyz, extra_xyz], dim=0)
            rgb = torch.cat([rgb, extra_rgb], dim=0)
            
    assert "facex" in cfg, "facex must be in cfg for point_e_initialize_with_bbox"
    # align the point cloud to the x axis
    console.print("[red]will align the point cloud to the x axis")
    x, y, z = xyz.chunk(3, dim=-1)
    xyz = torch.cat([-y, x, z], dim=-1)


    ############################## xyz ##############################
    def get_bbox(xyz):
        bbox = {}
        bbox["min_bound"], _ = torch.min(xyz, dim=0)
        bbox["max_bound"], _ = torch.max(xyz, dim=0)
        bbox["center"] = (bbox["min_bound"] + bbox["max_bound"]) / 2
        return bbox
    bbox = get_bbox(xyz)
    
    
    ### make xyz centered ### 
    xyz = xyz - bbox["center"]
    bbox["min_bound"] = bbox["min_bound"] - bbox["center"]
    bbox["max_bound"] = bbox["max_bound"] - bbox["center"]
    bbox["center"] = (bbox["min_bound"] + bbox["max_bound"]) / 2
    assert torch.allclose(
        bbox["center"], 
        torch.zeros(3, dtype=torch.float32, device=bbox["center"].device),
        atol=1e-6
        ), f"bbox center is not close to zero, {bbox['center']}"
    
    
    # scale the point cloud to fit the bbox_dimensions at x, y, z axis
    cur_dimensions = bbox["max_bound"] - bbox["min_bound"]
    if cfg.get("bbox_dimensions", None) is None:
        scale_factors = torch.ones(3, dtype=torch.float32, device=cur_dimensions.device)
    else:
        target_dimensions = torch.tensor(
            cfg.bbox_dimensions, # cfg.bbox_dimensions is a tuple
            dtype=torch.float32,
            device=cur_dimensions.device
            )
        scale_factors = target_dimensions / cur_dimensions

        assert cur_dimensions.shape == (3,), f"cur_dimensions shape is not (3,), {cur_dimensions.shape}"
        assert target_dimensions.shape == (3,), f"target_dimensions shape is not (3,), {target_dimensions.shape}"
        assert scale_factors.shape == (3,), f"scale_factors shape is not (3,), {scale_factors.shape}"
    xyz *= scale_factors
    xyz = xyz / (xyz.norm(dim=-1).max() + 1e-5)
    xyz = xyz * cfg.mean_std
    
    if cfg.get("bbox_dimensions", None) is None: 
        bbox = get_bbox(xyz)
        cur_dimensions = bbox["max_bound"] - bbox["min_bound"]
        OmegaConf.set_struct(cfg, False)
        cfg.bbox_dimensions = cur_dimensions.tolist()
        OmegaConf.set_struct(cfg, True)
        console.print("[red]cfg.bbox_dimensions = cur_dimensions.tolist()")
    ############################## xyz ##############################



    if cfg.get("random_color", False):
        console.print("[red]will use random color")
        rgb = torch.rand_like(rgb)

    if cfg.get("white_color", False):
        console.print("[red]will make all the gaussians white, for experimental usage")
        rgb = torch.ones_like(rgb) * 0.7

    assert "z_scale" not in cfg, "z_scale is not supported in point_e_initialize_with_bbox"

    initial_values = {}
    initial_values["mean"] = xyz
    initial_values["color"] = rgb
    # breakpoint()
    initial_values["svec"] = get_svec(cfg)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    return initial_values

def shap_e_initialize(cfg):
    if not hasattr(shap_e_initialize, 'initialized'):
        from utils.shap_e_helper import shap_e_generate_pcd_from_text
        shap_e_initialize.initialized = True
        shap_e_initialize.generate_pcd = shap_e_generate_pcd_from_text

    prompt = cfg.prompt
    pcd = shap_e_initialize.generate_pcd(prompt)
    
    ### Randomly select cfg.num_points from pcd ###
    if pcd.shape[0] > cfg.num_points:
        indices = torch.randperm(pcd.shape[0], device=pcd.device)[:cfg.num_points]
        pcd = pcd[indices]
    elif pcd.shape[0] < cfg.num_points:
        # If we have fewer points than required, we'll duplicate some randomly
        additional_indices = torch.randint(0, pcd.shape[0], (cfg.num_points - pcd.shape[0],), device=pcd.device)
        pcd = torch.cat([pcd, pcd[additional_indices]], dim=0)
    
    assert pcd.shape[0] == cfg.num_points, f"Expected {cfg.num_points} points, but got {pcd.shape[0]}"
    ### --------------------------------------- ###
    
    xyz, rgb = pcd[:, :3], pcd[:, 3:]

    xyz -= xyz.mean(dim=0, keepdim=True)

    xyz = xyz / (xyz.norm(dim=-1).max() + 1e-5)
    xyz = xyz * cfg.mean_std

    if cfg.get("facex", False):
        # align the point cloud to the x axis
        console.print("[red]will align the point cloud to the x axis")
        x, y, z = xyz.chunk(3, dim=-1)
        xyz = torch.cat([-y, x, z], dim=-1)

    if cfg.get("random_color", False):
        console.print("[red]will use random color")
        rgb = torch.rand_like(rgb)

    if cfg.get("white_color", False):
        console.print("[red]will make all the gaussians white, for experimental usage")
        rgb = torch.ones_like(rgb) * 0.7

    z_scale = cfg.get("z_scale", 1.0)
    xyz[..., 2] *= z_scale

    initial_values = {}
    initial_values["mean"] = xyz
    initial_values["color"] = rgb
    # breakpoint()
    initial_values["svec"] = get_svec(cfg)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    return initial_values


def shap_e_initialize_with_bbox(cfg):
    if not hasattr(shap_e_initialize_with_bbox, 'initialized'):
        from utils.shap_e_helper import shap_e_generate_pcd_from_text
        shap_e_initialize_with_bbox.initialized = True
        shap_e_initialize_with_bbox.generate_pcd = shap_e_generate_pcd_from_text

    prompt = cfg.prompt
    pcd = shap_e_initialize_with_bbox.generate_pcd(prompt)
    
    ### Randomly select cfg.num_points from pcd ###
    if pcd.shape[0] > cfg.num_points:
        indices = torch.randperm(pcd.shape[0], device=pcd.device)[:cfg.num_points]
        pcd = pcd[indices]
    elif pcd.shape[0] < cfg.num_points:
        # If we have fewer points than required, we'll duplicate some randomly
        additional_indices = torch.randint(0, pcd.shape[0], (cfg.num_points - pcd.shape[0],), device=pcd.device)
        pcd = torch.cat([pcd, pcd[additional_indices]], dim=0)
    
    assert pcd.shape[0] == cfg.num_points, f"Expected {cfg.num_points} points, but got {pcd.shape[0]}"
    ### --------------------------------------- ###
    
    xyz, rgb = pcd[:, :3], pcd[:, 3:]


    assert "facex" in cfg, "facex must be in cfg for shap_e_intialize_with_bbox"
    # align the point cloud to the x axis
    console.print("[red]will align the point cloud to the x axis")
    x, y, z = xyz.chunk(3, dim=-1)
    xyz = torch.cat([-y, x, z], dim=-1)


    ############################## xyz ##############################
    def get_bbox(xyz):
        bbox = {}
        bbox["min_bound"], _ = torch.min(xyz, dim=0)
        bbox["max_bound"], _ = torch.max(xyz, dim=0)
        bbox["center"] = (bbox["min_bound"] + bbox["max_bound"]) / 2
        return bbox
    bbox = get_bbox(xyz)
    
    
    ### make xyz centered ### 
    xyz = xyz - bbox["center"]
    bbox["min_bound"] = bbox["min_bound"] - bbox["center"]
    bbox["max_bound"] = bbox["max_bound"] - bbox["center"]
    bbox["center"] = (bbox["min_bound"] + bbox["max_bound"]) / 2
    assert torch.allclose(
        bbox["center"], 
        torch.zeros(3, dtype=torch.float32, device=bbox["center"].device)
        ), f"bbox center is not zero, {bbox['center']}"
    
    
    # scale the point cloud to fit the bbox_dimensions at x, y, z axis
    cur_dimensions = bbox["max_bound"] - bbox["min_bound"]
    if cfg.get("bbox_dimensions", None) is None:
        scale_factors = torch.ones(3, dtype=torch.float32, device=cur_dimensions.device)
    else:
        target_dimensions = torch.tensor(
            cfg.bbox_dimensions, # cfg.bbox_dimensions is a tuple
            dtype=torch.float32,
            device=cur_dimensions.device
            )
        scale_factors = target_dimensions / cur_dimensions

        assert cur_dimensions.shape == (3,), f"cur_dimensions shape is not (3,), {cur_dimensions.shape}"
        assert target_dimensions.shape == (3,), f"target_dimensions shape is not (3,), {target_dimensions.shape}"
        assert scale_factors.shape == (3,), f"scale_factors shape is not (3,), {scale_factors.shape}"
    xyz *= scale_factors
    xyz = xyz / (xyz.norm(dim=-1).max() + 1e-5)
    xyz = xyz * cfg.mean_std
    
    if cfg.get("bbox_dimensions", None) is None: 
        bbox = get_bbox(xyz)
        cur_dimensions = bbox["max_bound"] - bbox["min_bound"]
        OmegaConf.set_struct(cfg, False)
        cfg.bbox_dimensions = cur_dimensions.tolist()
        OmegaConf.set_struct(cfg, True)
        console.print("[red]cfg.bbox_dimensions = cur_dimensions.tolist()")
    ############################## xyz ##############################


    if cfg.get("random_color", False):
        console.print("[red]will use random color")
        rgb = torch.rand_like(rgb)

    if cfg.get("white_color", False):
        console.print("[red]will make all the gaussians white, for experimental usage")
        rgb = torch.ones_like(rgb) * 0.7

    assert "z_scale" not in cfg, "z_scale is not supported in shap_e_intialize_with_bbox"
    

    initial_values = {}
    initial_values["mean"] = xyz
    initial_values["color"] = rgb
    # breakpoint()
    initial_values["svec"] = get_svec(cfg)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    return initial_values

def point_cloud_initialize(cfg):
    initial_values = {}
    pcd = Path(cfg.pcd)
    assert pcd.exists(), f"point cloud file {pcd} does not exist"
    extension_name = pcd.suffix
    if extension_name == ".npy":
        pcd = torch.from_numpy(np.load(pcd))
    elif extension_name in [".pt", ".pth"]:
        pcd = torch.load(pcd)
    else:
        raise ValueError(f"Unknown point cloud file extension {extension_name}")

    xyz = pcd[:, :3]
    rgb = pcd[:, 3:]
    cfg.num_points = xyz.shape[0]
    num_points = xyz.shape[0]
    if cfg.svec_val > 0.0:
        svec = torch.ones(num_points, 3, dtype=torch.float32) * cfg.svec_val
    else:
        svec = nearest_neighbor_initialize(xyz, k=3)[..., None].repeat(1, 3)
    alpha = get_alpha(cfg)
    qvec = get_qvec(cfg)

    # xyz[..., 0], xyz[..., 1] = xyz[..., 1], xyz[..., 0]
    x, y, z = xyz.chunk(3, dim=-1)
    xyz = torch.cat([-y, x, z], dim=-1)

    # breakpoint()
    initial_values["mean"] = xyz
    initial_values["color"] = rgb
    initial_values["svec"] = svec
    initial_values["qvec"] = qvec
    initial_values["alpha"] = alpha
    initial_values["raw"] = False

    return initial_values


def mesh_initlization(cfg):
    mesh_path = Path(cfg.mesh)
    assert mesh_path.exists(), f"Mesh path {mesh_path} does not exist"
    xyz, rgb = load_mesh_as_pcd_trimesh(mesh_path, cfg.num_points)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    xyz -= xyz.mean(dim=0, keepdim=True)

    xyz = xyz / (xyz.norm(dim=-1).max() + 1e-5)
    xyz = xyz * cfg.mean_std

    if cfg.get("flip_yz", False):
        console.print("[red]will flip the y and z axis")
        x, y, z = xyz.chunk(3, dim=-1)
        xyz = torch.cat([x, z, y], dim=-1)

    if cfg.get("flip_xy", False):
        console.print("[red]will flip the x and y axis")
        x, y, z = xyz.chunk(3, dim=-1)
        xyz = torch.cat([y, x, z], dim=-1)

    if cfg.svec_val > 0.0:
        svec = get_svec(cfg)
    else:
        svec = nearest_neighbor_initialize(xyz, k=3)[..., None].repeat(1, 3)
    alpha = get_alpha(cfg)
    qvec = get_qvec(cfg)

    if cfg.get("random_color", True):
        rgb = torch.rand_like(rgb)

    initial_values = {}
    initial_values["mean"] = xyz
    initial_values["color"] = rgb
    initial_values["svec"] = svec
    initial_values["qvec"] = qvec
    initial_values["alpha"] = alpha
    initial_values["raw"] = False

    return initial_values


def from_ckpt(cfg):
    ckpt_path = Path(cfg.ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if cfg is None:
        cfg = {}
    if not "params" in ckpt:
        # case for loading only renderer ckpt
        new_cfg = OmegaConf.create(ckpt["cfg"])
        new_cfg.update(cfg)
        del ckpt["cfg"]
        cfg = new_cfg
    else:
        new_cfg = OmegaConf.create(ckpt["cfg"]).renderer
        new_cfg.update(cfg)
        ckpt = ckpt["params"]
    # This following two lines cause a bug when loading from ckpt, so I commented them out
    # if ckpt["color"].max() > 1 or ckpt["color"].min() < 0:
    #     ckpt["color"] = torch.sigmoid(ckpt["color"])
    ckpt["raw"] = True

    return ckpt


def image_initialize(cfg, **kwargs):
    # will generate 2 * cfg.num_points gaussian, half for front view, others will be optmized for front view
    num_points = cfg.num_points
    image = kwargs["image"].squeeze()
    # TODO: finish this
    depth_map = kwargs["depth_map"]
    c2w = kwargs["c2w"]
    camera_info = kwargs["camera_info"]
    mask = kwargs["mask"].squeeze()

    camera_info = CameraInfo.from_reso(depth_map.shape[1])
    pcd = lift_to_3d(depth_map, camera_info, c2w)
    pcd = pcd[mask]
    rgb = image[mask].to(pcd.device)
    print(pcd[..., 0].max())
    print(pcd[..., 0].min())
    print(pcd[..., 0].std())

    # breakpoint()
    if pcd.shape[0] > num_points:
        _, idx = farthest_point_sampling(pcd, num_points)
        # idx = idx.to(pcd.device)
        pcd = pcd[idx]
        rgb = rgb[idx]

    additional_pts = semisphere_initialize(cfg)

    cfg.num_points = pcd.shape[0]

    image_base_pts = {
        "mean": pcd,
        "color": rgb,
        "svec": get_svec(cfg),
        "qvec": get_qvec(cfg),
        "alpha": get_alpha(cfg),
    }

    initialize_values = {}
    for key in image_base_pts:
        initialize_values[key] = torch.cat(
            [image_base_pts[key], additional_pts[key]], dim=0
        )

    if cfg.get("grad_mask", False):
        grad_mask = torch.ones_like(initialize_values["mean"][..., 0])
        grad_mask[: pcd.shape[0]] = 0.0
        initialize_values["mask"] = grad_mask

    return initialize_values


def point_e_image_initialize(cfg, **kwargs):
    from utils.point_e_helper import point_e_generate_pcd_from_image

    if "image" in kwargs:
        image = kwargs["image"].squeeze()
    else:
        assert hasattr(cfg, "image"), "image not found in cfg"
        image = str(cfg.image)
    pcd = point_e_generate_pcd_from_image(
        image, cfg.num_points, cfg.get("base_name", None)
    )
    xyz, rgb = pcd[:, :3], pcd[:, 3:]
    xyz = xyz / (xyz.norm(dim=-1).max() + 1e-5)
    xyz = xyz * cfg.mean_std

    if cfg.get("facex", False):
        # align the point cloud to the x axis
        x, y, z = xyz.chunk(3, dim=-1)
        xyz = torch.cat([-y, x, z], dim=-1)

    initial_values = {}
    initial_values["mean"] = xyz
    initial_values["color"] = rgb
    # breakpoint()
    initial_values["svec"] = get_svec(cfg)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    return initial_values


def unbounded_initialize(cfg):
    R = cfg.mean_std
    N = cfg.num_points
    theta = torch.rand(N) * 2 * np.pi
    phi = torch.rand(N)
    phi = torch.acos(1 - 2 * phi)
    x = R * torch.sin(phi) * torch.cos(theta)
    y = R * torch.sin(phi) * torch.sin(theta)
    z = R * torch.cos(phi)

    initial_values = {}
    initial_values["mean"] = torch.stack([x, y, z], dim=1)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["svec"] = get_svec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    initial_values["color"] = torch.rand(cfg.num_points, 3, dtype=torch.float32)

    return initial_values


def box_initialize(cfg):
    L = cfg.mean_std
    N = cfg.num_points
    u = (torch.rand(N) * 2 - 1) * L
    v = (torch.rand(N) * 2 - 1) * L
    w = torch.ones_like(u) * L / 2
    w[::2] = -w[::2]
    xyz = torch.stack([u, v, w], dim=1)
    for i in range(N):
        permutations = torch.randperm(3)
        xyz[i] = xyz[i][permutations]

    initial_values = {}
    initial_values["mean"] = xyz
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["svec"] = get_svec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    initial_values["color"] = torch.rand(cfg.num_points, 3, dtype=torch.float32)

    return initial_values

from utils.autoencoder import OnehotAutoencoder
def initialize_OnehotAutoencoder(groupids, num_classes, num_epochs = 20, learning_rate = 0.01, batch_size = 2048, num_samples = 65536):
    # groupids: (num_points,) dtype=torch.int32
    class OnehotDataset(Dataset):
        def __init__(self, num_samples, num_classes, device):
            self.inputs, self.labels = self.generate_one_hot(num_samples, num_classes, device)
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, index):
            return self.inputs[index], self.labels[index]
        
        def generate_one_hot(self, num_samples, num_classes, device):
            labels = torch.randint(low=0, high=num_classes, size=(num_samples,), device=device)
            one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            return one_hot, labels
        
    
    model = OnehotAutoencoder(num_classes, device=groupids.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    dataset = OnehotDataset(num_samples, num_classes, groupids.device)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, drop_last=False,
                            generator=torch.Generator(device=groupids.device),
                            )
    
    model.train()
    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0
        total_loss = 0

        for inputs, labels in dataloader:
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(logits.detach(), 1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = total_loss / total_samples
        epoch_accuracy = 100 * total_correct / total_samples

        if (epoch + 1) % 5 == 0:
            console.print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
            
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def get_text_embeddings(texts):
    if not hasattr(get_text_embeddings, 'initialized'):
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        model_name = "openai/clip-vit-base-patch32"
        get_text_embeddings.model = CLIPTextModelWithProjection.from_pretrained(model_name)
        get_text_embeddings.tokenizer = AutoTokenizer.from_pretrained(model_name)
        get_text_embeddings.initialized = True

    inputs = get_text_embeddings.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = get_text_embeddings.model(**inputs)

    text_embeds = outputs.text_embeds
    return text_embeds

from utils.autoencoder import CLIPAutoencoder
def initialize_CLIPAutoencoder(
    groupids, num_classes, guidance_prompt_list,
    num_epochs = 500, learning_rate = 0.001, batch_size = 4,
    ):
    class TextEmbeddingDataset(torch.utils.data.Dataset):
        def __init__(self, embeddings):
            self.embeddings = embeddings

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            return self.embeddings[idx]
    

    prompts = [x.prompt for x in guidance_prompt_list]
    device = groupids.device
    
    high_dim_embeds = get_text_embeddings(prompts).to(device)
    model = CLIPAutoencoder(
        high_dim_embeds=high_dim_embeds, 
        encoding_dim=3,
    )
    dataset = TextEmbeddingDataset(model.high_dim_embeds_with_bg) # NOTE with bg
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=2, 
                                             shuffle=True, 
                                             drop_last=False,
                                             generator=torch.Generator(device=device),
                                             )
    
    criterion = CLIPAutoencoder.cal_clip_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print_period = num_epochs // 5
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Test the trained model
    model.eval()
    with torch.no_grad():
        results = model.classify(model.low_dim_embeds)
    
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def xyz2direction_ids(xyz):
    """
    xyz: (num_points, 3)
    """
    from utils.ops import (
        estimate_normal,
    )
    device = xyz.device
    normals = estimate_normal(xyz, neighborhood_size = 50, disambiguate_directions = True).to(device)
    # normals.shape == (num_points, 3)
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-5)
    
    def is_within_degree_threshold(normals, dir, degree_threshold):
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-5)
        dir = dir.to(torch.float32)
        dir = dir / torch.norm(dir)
        
        threshold_radians = torch.deg2rad(torch.tensor(degree_threshold, device=device))
        
        cos_threshold = torch.cos(threshold_radians)
        
        dot_products = torch.matmul(normals, dir)
        
        result = dot_products > cos_threshold
        
        return result # torch.Size([num_points])
    
    def is_front_view(normals, degree_threshold=45):
        normals_xy = normals[:, :2]
        front_view_mask = is_within_degree_threshold(
            normals_xy, torch.tensor([1, 0], device=device), degree_threshold
            )
        return front_view_mask
    
    def is_back_view(normals, degree_threshold=45):
        normals_xy = normals[:, :2]
        back_view_mask = is_within_degree_threshold(
            normals_xy, torch.tensor([-1, 0], device=device), degree_threshold
            )
        return back_view_mask
    
    def is_overhead_view(normals, degree_threshold=40):
        overhead_view_mask = is_within_degree_threshold(
            normals, torch.tensor([0, 0, 1], device=device), degree_threshold
            )
        return overhead_view_mask
    
    # 0 for side view, 1 for front view, 2 for back view, 3 for overhead view, follow the order in BasePromptProcessor
    direction_ids = torch.zeros(normals.shape[0], dtype=torch.int32, device=device) # torch.Size([num_points])
    front_view_mask = is_front_view(normals) # torch.Size([num_points])
    back_view_mask = is_back_view(normals)
    overhead_view_mask = is_overhead_view(normals)
    direction_ids[front_view_mask] = 1
    direction_ids[back_view_mask] = 2
    direction_ids[overhead_view_mask] = 3
    
    return direction_ids

def multiple_groups_initialize(cfg):
    device = cfg.device    
    mean_list = []
    qvec_list = []
    svec_list = []
    alpha_list =[]
    color_list = []
    groupid_list = []
    guidance_prompt_list = []
    # initialize object2world_qvec_for_groups, shape (cfg.init.num_groups, 4), wijk
    object2world_qvec_for_groups = torch.zeros(cfg.num_groups, 4, dtype=torch.float32, device=device) # zeros
    object2world_qvec_for_groups[:, 0] = 1.0
    
    object2world_scale_scalar_for_groups = torch.ones(cfg.num_groups, dtype=torch.float32, device=device)
    object2world_Tvec_for_groups = torch.zeros(cfg.num_groups, 3, dtype=torch.float32, device=device) # zeros
    restriction_bbox_max = torch.zeros(cfg.num_groups, 3, dtype=torch.float32, device=device)
    restriction_bbox_min = torch.zeros(cfg.num_groups, 3, dtype=torch.float32, device=device)
    local_object_center_of_groups = torch.zeros(cfg.num_groups, 3, dtype=torch.float32, device=device)
    object_groupids_list = []

    groupid = 0
    for subcfg in cfg.subinit:
        initial_values = initialize(subcfg)
        for k, v in initial_values.items():
            initial_values[k] = v.to(device)
        mean_list.append(initial_values['mean'])
        qvec_list.append(initial_values['qvec'])
        svec_list.append(initial_values['svec'])
        alpha_list.append(initial_values['alpha'])
        color_list.append(initial_values['color'])
        
        for p in subcfg.part_specific_guidance_prompts:
            guidance_prompt_list.append(p)
        
        bbox = {}
        bbox["min_bound"], _ = torch.min(initial_values['mean'], dim=0)
        bbox["max_bound"], _ = torch.max(initial_values['mean'], dim=0)
        
        
        hyperparameter_init_bbox_ratio = cfg.init_bbox_ratio
        if subcfg.get("init_bbox_ratio", None) is not None:
            hyperparameter_init_bbox_ratio = subcfg.init_bbox_ratio
            console.print(f"[green]### Override init_bbox_size: {hyperparameter_init_bbox_ratio}, for {subcfg.prompt}[/green]")
        else:
            console.print(f"[green]### Use default init_bbox_size: {hyperparameter_init_bbox_ratio}, for {subcfg.prompt}[/green]")
        
        
        hyperparameter_restriction_bbox_ratio = cfg.restriction_bbox_ratio
        if subcfg.get("restriction_bbox_ratio", None) is not None:
            hyperparameter_restriction_bbox_ratio = subcfg.restriction_bbox_ratio
            console.print(f"[green]### Override restriction_bbox_size: {hyperparameter_restriction_bbox_ratio}, for {subcfg.prompt}[/green]")
        else:
            console.print(f"[green]### Use default restriction_bbox_size: {hyperparameter_restriction_bbox_ratio}, for {subcfg.prompt}[/green]")
        
        if cfg.restriction_bbox_type == "centered_on_object":
            bbox["init_min_bound"] = bbox["min_bound"]
            bbox["init_max_bound"] = bbox["max_bound"]
            bbox["center"] = (bbox["init_min_bound"] + bbox["init_max_bound"]) / 2
            bbox["init_size_under_local_coord"] = bbox["init_max_bound"] - bbox["init_min_bound"]
            bbox["restriction_size_under_local_coord"] = bbox["init_size_under_local_coord"] * (
                hyperparameter_restriction_bbox_ratio / hyperparameter_init_bbox_ratio)
            bbox["restriction_max_bound"] = bbox["center"] + 0.5 * bbox["restriction_size_under_local_coord"]
            bbox["restriction_min_bound"] = bbox["center"] - 0.5 * bbox["restriction_size_under_local_coord"]
        
        
        groupid_tensor = torch.zeros(initial_values['mean'].shape[0], 
                                     dtype=torch.int32, device=initial_values['mean'].device)
        groupid_assigned_mask = torch.zeros(initial_values['mean'].shape[0]
                                            , dtype=torch.int32, device=initial_values['mean'].device)
        object_groupids = [] # contain the groupid for each part of the object
        for part_space_ratio in subcfg.part_space_ratios:
            part_space_ratio = torch.tensor(part_space_ratio, dtype=torch.float32, device=initial_values['mean'].device)
            # subregion.shape == (2,3)
            sub_bbox = {}
            sub_bbox["min_bound"] = bbox["min_bound"] + part_space_ratio[0] * (bbox["max_bound"] - bbox["min_bound"])
            sub_bbox["max_bound"] = bbox["min_bound"] + part_space_ratio[1] * (bbox["max_bound"] - bbox["min_bound"])
            # Get a mask (num_points,) for the points in the subregion
            # initial_values['mean'].shape == (num_points, 3)
            # sub_bbox["min_bound"].shape == (3,)
            subregion_mask = (initial_values['mean'] >= sub_bbox["min_bound"]) & (initial_values['mean'] <= sub_bbox["max_bound"])
            subregion_mask = subregion_mask.all(dim=1)
            
            ##############################################
            ####              groupid                 ####
            ##############################################
            # assign groupid to the points in the subregion
            groupid_tensor[subregion_mask] = groupid
            groupid_assigned_mask = groupid_assigned_mask | subregion_mask
            
            ##############################################
            ####           object_groupids            ####
            ##############################################
            object_groupids.append(groupid)
        
            ##############################################
            ####     local_object_center_of_groups    ####
            ##############################################
            local_object_center_of_groups[groupid] = (bbox["min_bound"] + bbox["max_bound"]) / 2

        
            ##############################################
            ####           object2world_qvec          ####
            ####           object2world_Tvec          ####
            ##############################################
            object2world_Tvec_for_groups[groupid] = torch.tensor(subcfg.xyz_offset, dtype=torch.float32, device=initial_values['mean'].device)
            
            
            ##############################################
            #### object2world_scale_scalar_for_groups ####
            ################################################################################################
            # L2G means local to global
            # Define: expected_size_under_global_coord = expected_size_under_local_coord * L2G
            # init_size_under_local_coord = expected_size_under_local_coord * hyperparameter_init_bbox_ratio
            # init_size_under_global_coord = init_size_under_local_coord * L2G
            # So, init_size_under_global_coord = expected_size_under_local_coord * L2G * hyperparameter_init_bbox_ratio
            ################################################################################################
            # expected_size_under_global_coord / init_size_under_local_coord
            #       = expected_size_under_global_coord / (expected_size_under_local_coord * hyperparameter_init_bbox_ratio)
            #       = L2G / hyperparameter_init_bbox_ratio
            # So, L2G_over_init_bbox_ratio means (L2G / hyperparameter_init_bbox_ratio)
            # So, L2G = L2G_over_init_bbox_ratio * hyperparameter_init_bbox_ratio
            # object2world_scale_scalar means L2G
            # So, object2world_scale_scalar = L2G_over_init_bbox_ratio * hyperparameter_init_bbox_ratio
            ################################################################################################
            init_size_under_local_coord = bbox["max_bound"] - bbox["min_bound"]
            expected_size_under_global_coord = torch.tensor(
                subcfg.bbox_dimensions,
                dtype=torch.float32,
                device=init_size_under_local_coord.device
                )

            L2G_over_init_bbox_ratio = expected_size_under_global_coord / init_size_under_local_coord
            assert torch.allclose(L2G_over_init_bbox_ratio, L2G_over_init_bbox_ratio[0] * torch.ones_like(L2G_over_init_bbox_ratio)), f"L2G_over_init_bbox_ratio is not the same, {L2G_over_init_bbox_ratio}"
            
            object2world_scale_scalar_for_groups[groupid] = torch.tensor(
                L2G_over_init_bbox_ratio[0].item() * hyperparameter_init_bbox_ratio,
                dtype=torch.float32,
                device=initial_values['mean'].device,
                ) # use .item() to prevent: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
            
            ####################################################
            #### restriction_bbox_max, restriction_bbox_min ####
            ####################################################
            #### we calculate the restriction_bbox_loss under local coordinate
            # restriction_size_under_local_coord = expected_size_under_local_coord * hyperparameter_restriction_bbox_ratio
            # init_size_under_local_coord = expected_size_under_local_coord * hyperparameter_init_bbox_ratio
            # So, restriction_size_under_local_coord 
            #       = expected_size_under_local_coord * hyperparameter_restriction_bbox_ratio
            #       = (init_size_under_local_coord / hyperparameter_init_bbox_ratio) * hyperparameter_restriction_bbox_ratio
            #       = init_size_under_local_coord * (hyperparameter_restriction_bbox_ratio / hyperparameter_init_bbox_ratio)
            ################################################################################################  
            # restriction_size_under_local_coord = init_size_under_local_coord * (
            #     hyperparameter_restriction_bbox_ratio / hyperparameter_init_bbox_ratio)
            if cfg.restriction_bbox_type == "centered_on_part":
                sub_bbox["center"] = (sub_bbox["min_bound"] + sub_bbox["max_bound"]) / 2
                sub_bbox["init_size_under_local_coord"] = sub_bbox["max_bound"] - sub_bbox["min_bound"]
                sub_bbox["restriction_size_under_local_coord"] = sub_bbox["init_size_under_local_coord"] * (
                    hyperparameter_restriction_bbox_ratio / hyperparameter_init_bbox_ratio)
                sub_bbox["restriction_max_bound"] = sub_bbox["center"] + 0.5 * sub_bbox["restriction_size_under_local_coord"]
                sub_bbox["restriction_min_bound"] = sub_bbox["center"] - 0.5 * sub_bbox["restriction_size_under_local_coord"]
                restriction_bbox_max[groupid] = sub_bbox["restriction_max_bound"]
                restriction_bbox_min[groupid] = sub_bbox["restriction_min_bound"]
            elif cfg.restriction_bbox_type == "centered_on_object":
                sub_bbox["init_min_bound"] = sub_bbox["min_bound"]
                sub_bbox["init_max_bound"] = sub_bbox["max_bound"]
                
                mask = (part_space_ratio[1] == 1)
                restriction_bbox_max[groupid] = mask * bbox["restriction_max_bound"] + (~mask) * sub_bbox["init_max_bound"]
                mask = (part_space_ratio[0] == 0)
                restriction_bbox_min[groupid] = mask * bbox["restriction_min_bound"] + (~mask) * sub_bbox["init_min_bound"]
                mask = None
            else:
                raise NotImplementedError(f"Unknown restriction_bbox_type: {cfg.restriction_bbox_type}")

            ##############################################
            ####              groupid += 1            ####
            ##############################################
            groupid += 1
        
        assert torch.all(groupid_assigned_mask), f"some points are not assigned to any group"
        
        groupid_list.append(
            groupid_tensor
        )
        object_groupids_list.append(object_groupids)
    
    assert groupid == cfg.num_groups, f"num_groups is not equal to the number of groups"

        
    # catenate all the groups
    initial_values = {}
    initial_values["mean"] = torch.cat(mean_list, dim=0).to(cfg.device)
    initial_values["qvec"] = torch.cat(qvec_list, dim=0).to(cfg.device)
    initial_values["svec"] = torch.cat(svec_list, dim=0).to(cfg.device)
    initial_values["alpha"] = torch.cat(alpha_list, dim=0).to(cfg.device)
    initial_values["color"] = torch.cat(color_list, dim=0).to(cfg.device)
    initial_values["groupid"] = torch.cat(groupid_list, dim=0).to(cfg.device)
    assert initial_values["mean"].shape[0] == initial_values["groupid"].shape[0]
    initial_values["object2world_qvec_for_groups"] = object2world_qvec_for_groups.to(cfg.device)
    initial_values["object2world_scale_scalar_for_groups"] = object2world_scale_scalar_for_groups.to(cfg.device)
    initial_values["object2world_Tvec_for_groups"] = object2world_Tvec_for_groups.to(cfg.device)
    initial_values["restriction_bbox_max"] = restriction_bbox_max.to(cfg.device)
    initial_values["restriction_bbox_min"] = restriction_bbox_min.to(cfg.device)
    initial_values["local_object_center_of_groups"] = local_object_center_of_groups.to(cfg.device)
    
    if cfg.autoencoder_type == "OnehotAutoencoder":
        initial_values["autoencoder"] = initialize_OnehotAutoencoder(
            groupids=initial_values["groupid"], 
            num_classes=cfg.num_groups + 1
        )
    elif cfg.autoencoder_type == "CLIPAutoencoder":
        initial_values["autoencoder"] = initialize_CLIPAutoencoder(
            groupids=initial_values["groupid"], 
            num_classes=cfg.num_groups + 1,
            guidance_prompt_list=guidance_prompt_list,
        )
    else:
        raise ValueError(f"Unknown autoencoder_type: {cfg.autoencoder_type}")
    
    initial_values["guidance_prompts"] = guidance_prompt_list
    initial_values["object_groupids_list"] = object_groupids_list
    
    return initial_values

def initialize(cfg, **kwargs):
    init_type = cfg.type
    if init_type == "base":
        return base_initialize(cfg)
    elif init_type == "unisphere":
        return unisphere_initialize(cfg)
    elif init_type == "point_e":
        return point_e_initialize(cfg)
    elif init_type == "point_e_with_bbox":
        return point_e_initialize_with_bbox(cfg)
    elif init_type == "shap_e":
        return shap_e_initialize(cfg)
    elif init_type == "shap_e_with_bbox":
        return shap_e_initialize_with_bbox(cfg)
    elif init_type == "ckpt":
        return from_ckpt(cfg)
    elif init_type == "image":
        return image_initialize(cfg, **kwargs)
    elif init_type == "point_cloud":
        return point_cloud_initialize(cfg, **kwargs)
    elif init_type == "mesh":
        return mesh_initlization(cfg, **kwargs)
    elif init_type == "point_e_image":
        return point_e_image_initialize(cfg, **kwargs)
    elif init_type == "unbounded":
        return unbounded_initialize(cfg)
    elif init_type == "box":
        return box_initialize(cfg)
    elif init_type == "multiple_groups":
        return multiple_groups_initialize(cfg) # NOTE can be only called once, not recursively
    else:
        raise NotImplementedError(f"Unknown initialization type: {init_type}")
