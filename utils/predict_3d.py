import os
import json

import torch
import numpy as np
import open3d as o3d

from utils.util_traj import traj2vec
from utils.recon_3d import recon_3d_mast3r, recon_3d_dust3r
from utils.make_trajectory import generate_smooth_camera_path

def sample_and_pad_3d(data, size=4096):
    if len(data) > size:
        indices = np.random.choice(len(data), size, replace=False)
        data = data[indices]
    elif len(data) < size:
        pad_size = size - len(data)
        pad_indices = np.random.choice(len(data), pad_size, replace=True)
        data = np.concatenate([data, data[pad_indices]], axis=0)
        indices = np.concatenate([np.arange(len(data)), pad_indices], axis=0)
    return data, indices

def sample_and_pad_trajectory(data, size=100):
    if len(data) > size:
        indices = np.random.choice(len(data)-1, size-2, replace=False) + 1
        data = np.concatenate([data[0:1], data[indices], data[-1:]], axis=0)
    elif len(data) < size:
        pad_value = data[-1]
        pad_size = size - len(data)
        data = np.concatenate([data, np.repeat(pad_value.reshape(1,-1), pad_size, axis=0)], axis=0)
        indices = np.concatenate([np.arange(len(data)), np.full(pad_size, len(data)-1)], axis=0)
    return data, indices

def get_extrinsics_intrinsics(img_dir):
    extrinsics = None
    intrinsics = None

    extrinsics_path = os.path.join(img_dir, 'extrinsics.json')
    intrinsics_path = os.path.join(img_dir, 'intrinsics.json')
    
    if os.path.exists(extrinsics_path):
        with open(extrinsics_path, 'r') as f:
            data_extrinsics = json.load(f)

        extrinsics = torch.tensor(data_extrinsics['extrinsics']).float() # (2, 4, 4)

    if os.path.exists(intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            data_intrinsics = json.load(f)

        intrinsics = {
            "focals": data_intrinsics['focals'], # len 2 list
            "principal_points": torch.tensor(data_intrinsics['principal_points']).float() # (2, 2)
        }    

    return extrinsics, intrinsics

def get_trajectory(img_dir, poses, length = 20):
    trajectory_path = os.path.join(img_dir, 'trajectory.json')

    if os.path.exists(trajectory_path):
        with open(trajectory_path, 'r') as f:
            data_trajectory = json.load(f)

        trajectory = np.array(data_trajectory['trajectory']) # (N, 4, 4)
    else:
        assert poses is not None
        first_pose = poses[0].detach().cpu().numpy()
        last_pose = poses[-1].detach().cpu().numpy()
        positions, rotations = generate_smooth_camera_path(first_pose[:3,:3], first_pose[:3,3], last_pose[:3,:3], last_pose[:3,3], length)
        trajectory = []
        for i in range(len(positions)):
            R = np.array(rotations[i])
            T = np.array(positions[i]).reshape(-1,1)
            matrix = np.concatenate((np.concatenate((R, T), axis=1), np.array([[0,0,0,1]])), axis=0)
            trajectory.append(matrix)
        trajectory = np.array(trajectory)

        data_trajectory = {
            "trajectory": trajectory.tolist()
        }
        with open(trajectory_path, 'w') as f:
            json.dump(data_trajectory, f)

    return trajectory

def get_guidance_input(img_path, use_mast3r=True):
    extrinsics, intrinsics = get_extrinsics_intrinsics(img_path)
    if use_mast3r:
        _, _, poses, _, pts3d, _, confidence_masks, _, _ = recon_3d_mast3r(img_path, extrinsics, intrinsics)
    else:
        _, _, poses, _, pts3d, _, confidence_masks, _, _ = recon_3d_mast3r(img_path, extrinsics, intrinsics)
    trajectory = traj2vec(get_trajectory(img_path, poses))

    # tensor to numpy
    confidence_mask0 = confidence_masks[0].detach().cpu().numpy()
    confidence_mask1 = confidence_masks[1].detach().cpu().numpy()
    print(f"Confidence points: {np.sum(confidence_mask0) + np.sum(confidence_mask1)}")

    # change (H,W,3) to (N,3)
    pts3d1 = pts3d[0].detach().cpu().numpy()[confidence_mask0].reshape(-1,3)
    pts3d2 = pts3d[1].detach().cpu().numpy()[confidence_mask1].reshape(-1,3)

    pts3d1_sample, _ = sample_and_pad_3d(pts3d1, size=2048)
    pts3d2_sample, _ = sample_and_pad_3d(pts3d2, size=2048)
    pts3d_sample = np.concatenate([pts3d1_sample, pts3d2_sample], axis=0)

    return pts3d_sample, trajectory


def predict_3d(model, args, use_mast3r=True):
    extrinsics, intrinsics = get_extrinsics_intrinsics(args.img_path)

    if use_mast3r:
        imgs, focals, poses, pps, pts3d, conf, confidence_masks, matches_im0, matches_im1 = recon_3d_mast3r(args.img_path, extrinsics, intrinsics)
    else:
        imgs, focals, poses, pps, pts3d, conf, confidence_masks, matches_im0, matches_im1 = recon_3d_dust3r(args.img_path, extrinsics, intrinsics)

    trajectory = traj2vec(get_trajectory(args.img_path, poses))
    trajectory_tensor = torch.from_numpy(trajectory).float().to(model.trajectory_guidance_model.device)

    # tensor to numpy
    confidence_mask0 = confidence_masks[0].detach().cpu().numpy()
    confidence_mask1 = confidence_masks[1].detach().cpu().numpy()
    print(f"Confidence points: {np.sum(confidence_mask0) + np.sum(confidence_mask1)}")

    # save confidence masks and points
    confidence_mask0_path = os.path.join(args.img_path, 'confidence_mask0.npy')
    confidence_mask1_path = os.path.join(args.img_path, 'confidence_mask1.npy')
    np.save(confidence_mask0_path, confidence_mask0)
    np.save(confidence_mask1_path, confidence_mask1)

    pts3d0_path = os.path.join(args.img_path, 'pts3d0.npy')
    pts3d1_path = os.path.join(args.img_path, 'pts3d1.npy')
    np.save(pts3d0_path, pts3d[0].detach().cpu().numpy())
    np.save(pts3d1_path, pts3d[1].detach().cpu().numpy())

    # change (H,W,3) to (N,3)
    pts3d1 = pts3d[0].detach().cpu().numpy()[confidence_mask0].reshape(-1,3)
    pts3d2 = pts3d[1].detach().cpu().numpy()[confidence_mask1].reshape(-1,3)

    pts3d1_sample, indices1 = sample_and_pad_3d(pts3d1, size=2048)
    pts3d2_sample, indices2 = sample_and_pad_3d(pts3d2, size=2048)
    pts3d_sample = np.concatenate([pts3d1_sample, pts3d2_sample], axis=0)
    pts3d_sample_tensor = torch.from_numpy(pts3d_sample).float().to(model.spatial_guidance_model.device)

    pts3d_sample_tensor.to(model.spatial_guidance_model.device)
    trajectory_tensor.to(model.trajectory_guidance_model.device)

    guidance_3d = model.spatial_guidance_model(pts3d_sample_tensor)
    guidance_traj = model.trajectory_guidance_model(trajectory_tensor)
    print(f"Guidance 3D: {guidance_3d.shape}, Guidance Trajectory: {guidance_traj.shape}")

    return guidance_3d, guidance_traj, focals, pps, poses, pts3d, conf

    match_mask0 = np.zeros((confidence_mask0.shape[0], confidence_mask0.shape[1]))
    match_mask1 = np.zeros((confidence_mask1.shape[0], confidence_mask1.shape[1]))
    for i in range(matches_im0.shape[0]):
        match_mask0[matches_im0[i,1]][matches_im0[i,0]] = 1
    for i in range(matches_im1.shape[0]):
        match_mask1[matches_im1[i,1]][matches_im1[i,0]] = 1
    match_mask0 = match_mask0.astype(bool)
    match_mask1 = match_mask1.astype(bool)

    match_conf_mask0 = confidence_mask0 * match_mask0
    match_conf_mask1 = confidence_mask1 * match_mask1
    print(f"Matched points: {np.sum(match_mask0) + np.sum(match_mask1)}")
    print(f"Matched confidence points: {np.sum(match_conf_mask0) + np.sum(match_conf_mask1)}")


    # pts3d1_color = imgs[0][confidence_mask0]
    # pts3d2_color = imgs[1][confidence_mask1]
    # pts3d1_color_sample = pts3d1_color.reshape(-1,3)[indices1]
    # pts3d2_color_sample = pts3d2_color.reshape(-1,3)[indices2]
    # pts3d_color_sample = np.concatenate([pts3d1_color_sample, pts3d2_color_sample], axis=0)

    # demo code for visualization
    pts3d1 = pts3d[0].detach().cpu().numpy()[confidence_mask0]
    pts3d2 = pts3d[1].detach().cpu().numpy()[confidence_mask1]
    pts3d1_color = imgs[0][confidence_mask0]
    pts3d2_color = imgs[1][confidence_mask1]

    # pts3d1_match = pts3d[0].detach().cpu().numpy()[match_mask0]
    # pts3d2_match = pts3d[1].detach().cpu().numpy()[match_mask1]
    # pts3d1_match_color = imgs[0][match_mask0]
    # pts3d2_match_color = imgs[1][match_mask1]

    # pts3d1_match_conf = pts3d[0].detach().cpu().numpy()[match_conf_mask0]
    # pts3d2_match_conf = pts3d[1].detach().cpu().numpy()[match_conf_mask1]
    # pts3d1_match_conf_color = imgs[0][match_conf_mask0]
    # pts3d2_match_conf_color = imgs[1][match_conf_mask1]

    # pts3d1_match = pts3d1_match.reshape(-1,3)
    # pts3d2_match = pts3d2_match.reshape(-1,3)
    # pts3d1_match_color = pts3d1_match_color.reshape(-1,3)
    # pts3d2_match_color = pts3d2_match_color.reshape(-1,3)

    # pts3d1_match_conf = pts3d1_match_conf.reshape(-1,3)
    # pts3d2_match_conf = pts3d2_match_conf.reshape(-1,3)
    # pts3d1_match_conf_color = pts3d1_match_conf_color.reshape(-1,3)
    # pts3d2_match_conf_color = pts3d2_match_conf_color.reshape(-1,3)

    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts3d1)
    pcd2.points = o3d.utility.Vector3dVector(pts3d2)
    pcd1.colors = o3d.utility.Vector3dVector(pts3d1_color)
    pcd2.colors = o3d.utility.Vector3dVector(pts3d2_color)

    pcd1_sample = o3d.geometry.PointCloud()
    pcd2_sample = o3d.geometry.PointCloud()
    pcd1_sample.points = o3d.utility.Vector3dVector(pts3d1_sample)
    pcd2_sample.points = o3d.utility.Vector3dVector(pts3d2_sample)
    pcd1_sample.colors = o3d.utility.Vector3dVector(pts3d1_color_sample)
    pcd2_sample.colors = o3d.utility.Vector3dVector(pts3d2_color_sample)

    # pcd1_match = o3d.geometry.PointCloud()
    # pcd2_match = o3d.geometry.PointCloud()
    # pcd1_match.points = o3d.utility.Vector3dVector(pts3d1_match)
    # pcd2_match.points = o3d.utility.Vector3dVector(pts3d2_match)
    # pcd1_match.colors = o3d.utility.Vector3dVector(pts3d1_match_color)
    # pcd2_match.colors = o3d.utility.Vector3dVector(pts3d2_match_color)

    # pcd1_match_conf = o3d.geometry.PointCloud()
    # pcd2_match_conf = o3d.geometry.PointCloud()
    # pcd1_match_conf.points = o3d.utility.Vector3dVector(pts3d1_match_conf)
    # pcd2_match_conf.points = o3d.utility.Vector3dVector(pts3d2_match_conf)
    # pcd1_match_conf.colors = o3d.utility.Vector3dVector(pts3d1_match_conf_color)
    # pcd2_match_conf.colors = o3d.utility.Vector3dVector(pts3d2_match_conf_color)

    # o3d.visualization.draw_geometries([pcd1, pcd2])
    # o3d.visualization.draw_geometries([pcd1_sample, pcd2_sample])
    # o3d.visualization.draw_geometries([pcd1_match, pcd2_match])
    # o3d.visualization.draw_geometries([pcd1_match_conf, pcd2_match_conf])

    return None