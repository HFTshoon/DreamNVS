import os
import random
import argparse

import numpy as np
import open3d as o3d

from util_co3d import load_CO3D_data
from util_traj import rotmat2qvec

def get_camera_frustum(img_size, hfov, vfov, C2W, frustum_length=1, color=[0., 1., 0.]):
    W, H = img_size
    hfov_deg = np.rad2deg(hfov)
    vfov_deg = np.rad2deg(vfov)
    # print("hfov", hfov, vfov)
    half_w = frustum_length * np.tan(np.deg2rad(hfov_deg / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov_deg / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    #print("frustum_points afters", frustum_points)
    return frustum_points, frustum_lines, frustum_colors

def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def visualize_trajectory(seq_info, resolution=1, relative=False, verbose=False):
    all_poses =[]
    all_Ts = []
    trajectory = []
    seq_len = len(seq_info.seq_cameras)

    if verbose:
        print(f"Sequence length: {seq_len}")
        # print(f"Camera FovX/FovY (rad): {seq_info.seq_cameras[0].FovX}, {seq_info.seq_cameras[0].FovY}")
        print(f"Camera FovX/FovY (degree): {round(np.degrees(seq_info.seq_cameras[0].FovX),2)}, {round(np.degrees(seq_info.seq_cameras[0].FovY),2)}")

    for idx, camera_info in enumerate(seq_info.seq_cameras):
        R = camera_info.R
        T = camera_info.T
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        pose = np.linalg.inv(pose) # R, T -> R^T, -R^T * T

        all_Ts.append(pose[:3, 3])
        qvec = rotmat2qvec(pose[:3, :3])
        pose_qvec = np.concatenate([qvec, pose[:3, 3]])
        trajectory.append(pose_qvec)

        FOR_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.array([0., 0., 0.]))
        FOR_cam.transform(pose)
        all_poses.append(FOR_cam)
        frustums = []
        img_size = (camera_info.width/resolution, camera_info.height/resolution)
        frustums.append(get_camera_frustum(img_size, camera_info.FovX, camera_info.FovY, pose, frustum_length=0.5, color=[(seq_len-idx)/seq_len,0,idx/seq_len]))

        cameras = frustums2lineset(frustums)
        # PLOT CAMERAS HEREEEE
        all_poses.append(cameras)

    #draw coordinate frame at center and a sphere
    FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
    all_poses.append(FOR)

    #sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=5)
    #sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    #all_poses.append(sphere)

    #add lines along the trajectory
    lines = []
    for i in range(seq_len-1):
        points = np.array([all_Ts[i], all_Ts[i+1]])
        lines.append(points)
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.concatenate(lines, axis=0))
    lineset.lines = o3d.utility.Vector2iVector(np.array([[i, i+1] for i in range(0, len(lineset.points)-1, 2)]))
    all_poses.append(lineset)

    if verbose:
        for vec in trajectory:
            print("\t".join([str(round(x,3)) for x in vec]))
    o3d.visualization.draw_geometries(all_poses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/mydata/data/hyunsoo/co3d_sample_preprocess')
    parser.add_argument('--object', type=str, default='apple')
    parser.add_argument('--seq_num', type=int, default=-1)
    parser.add_argument('--seq_name', type=str, default='')
    parser.add_argument('--random_obj', action='store_true')
    parser.add_argument('--random_seq', action='store_true')
    parser.add_argument('--random_all',  action='store_true')
    args = parser.parse_args()

    data = load_CO3D_data(args.base_dir)

    object_list = data["object_list"]
    if args.random_all:
        args.random_obj = True
        args.random_seq = True

    if args.random_obj:
        args.object = random.choice(object_list)
    
    
    seq_list = data[args.object]["seq_name_list"]

    if args.random_seq or (args.seq_name == '' and args.seq_num == -1):
        args.seq_name = random.choice(seq_list)
    if args.seq_num != -1:
        args.seq_name = seq_list[min(args.seq_num, len(seq_list)-1)]
    
    seq_info = object_info = data[args.object][args.seq_name]
    visualize_trajectory(seq_info, resolution=1, verbose=True)
    
    
        