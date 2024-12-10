import os
import json

from tqdm import tqdm
import numpy as np
import torch

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

from preprocess.util_traj import fov2focal

def get_dust3r_model(model_path, device):
    return AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

def get_CO3D_dust3r_scene(model, cam_infos, sample_n):
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    image_list = []
    start_idx = int(cam_infos[0].image_name[-6:])
    end_idx = int(cam_infos[-1].image_name[-6:])

    # poses = []
    # focals = []
    # pps = []

    for cam_info in cam_infos:
        image_list.append(cam_info.image_path)
        # focals.append([fov2focal(cam_info.FovX, cam_info.width), fov2focal(cam_info.FovY, cam_info.height)])
        # pps.append(cam_info.pps)
        # pose = np.eye(4)
        # pose[:3, :3] = cam_info.R.T
        # pose[:3, 3] = cam_info.T
        # pose = np.diag([-1,-1,1,1]).astype(np.float32) @ pose
        # pose = np.linalg.inv(pose)
        # poses.append(pose)

    # poses = torch.tensor(poses)
    # focals = np.mean(focals, axis=0)
    # pps = torch.tensor(pps)
            
    # load_images can take a list of images or a directory
    images = load_images(image_list, size=512)
    pairs = make_pairs(images, scene_graph='swin', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

    # scene.preset_pose(poses)
    # scene.preset_focal(focals)
    # scene.preset_principal_point(pps)    

    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = to_numpy(scene.get_pts3d())
    confidence_masks = to_numpy(scene.get_masks())
    
    n = 4096
    pts_all = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])

    # sample n points
    print(pts_all.shape, start_idx, end_idx)
    if pts_all.shape[0] >= n:
        sample_idx = np.random.choice(pts_all.shape[0], n, replace=False)
    else:
        sample_idx = np.random.choice(pts_all.shape[0], n, replace=True)
    pts_all_sample = pts_all[sample_idx]
    return image_list, pts_all_sample, poses, focals, start_idx, end_idx
    
