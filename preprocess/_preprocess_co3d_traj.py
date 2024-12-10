import os
import pickle
import argparse

import numpy as np
from tqdm import tqdm

from util_co3d import read_CO3D_scene_info, get_CO3D_seq_info
from util_data import SeqInfo

def filter_seq_info(seq_info, iqr_factor=3, verbose=False):
    camera_width = seq_info.seq_cameras[0].width
    camera_height = seq_info.seq_cameras[0].height
    for camera in seq_info.seq_cameras:
        if (camera.width * camera.height) / (camera_height * camera_width) > 2 or (camera.width * camera.height) / (camera_height * camera_width) < 0.5:
            # if verbose:
            #     print(f"camera width/height mismatch: {camera_width}x{camera_height} vs {camera.width}x{camera.height}")
            return SeqInfo(seq_cameras=[], average_distance=-1)

    T_list = []
    for camera in seq_info.seq_cameras:
        pose = np.eye(4)
        pose[:3, :3] = camera.R
        pose[:3, 3] = camera.T
        pose = np.linalg.inv(pose)
        T_list.append(pose[:3, 3])

    T_diff_list = []
    for idx in range(len(seq_info.seq_cameras)-1):
        before_T = T_list[idx]
        after_T = T_list[idx+1]
        T_diff = after_T - before_T
        T_diff_norm = np.linalg.norm(T_diff)

        T_diff_list.append((T_diff_norm, idx))

    # get the median and std of T_diff
    T_diff_list.sort(key=lambda x: x[0])
    q25_T_diff = T_diff_list[len(T_diff_list)//4][0]
    q75_T_diff = T_diff_list[3*len(T_diff_list)//4][0]
    iqr_T_diff = q75_T_diff - q25_T_diff
    upper_bound_T_diff = q75_T_diff + iqr_factor * iqr_T_diff

    # filter out the seq_cameras that have T_diff larger than upper_bound_T_diff
    far_idx = []
    T_diff_list.sort(key=lambda x: x[1])
    for T_diff_norm, idx in T_diff_list:
        if T_diff_norm > upper_bound_T_diff:
            far_idx.append(idx)
    far_idx.append(len(seq_info.seq_cameras)-1)
    far_idx.sort()

    biggest_seq = (-1, -1)
    st = 0
    for idx in far_idx:
        if idx - st > biggest_seq[1] - biggest_seq[0]:
            biggest_seq = (st, idx)
        st = idx + 1

    # if verbose:
    #     print(f"biggest_seq: {biggest_seq}, len: {len(seq_info.seq_cameras)} -> {biggest_seq[1]-biggest_seq[0]+1}")
    refined_seq_cameras = seq_info.seq_cameras[biggest_seq[0]:biggest_seq[1]+1]
    refined_average_distance = np.mean([np.linalg.norm(camera.T) for camera in refined_seq_cameras])
    refined_seq_info = SeqInfo(seq_cameras=refined_seq_cameras, average_distance=refined_average_distance)
    return refined_seq_info

def preprocess_CO3D_object(base_dir, preprocess_dir, object, iqr_factor=3, not_filter=False, verbose=False):
    scene_path = os.path.join(base_dir, object)
    scene_info = read_CO3D_scene_info(scene_path)

    preprocess_path = os.path.join(preprocess_dir, object)
    if not os.path.exists(preprocess_path):
        os.makedirs(preprocess_path)
    object_seq_info_path = os.path.join(preprocess_path, f"object_seq_info.pkl")
    object_seq_name_txt_path = os.path.join(preprocess_path, f"object_seq_name.txt")

    seq_name_list = os.listdir(scene_path)
    seq_name_list = [name for name in seq_name_list if os.path.isdir(os.path.join(scene_path, name))]
    seq_name_list.sort()

    failed = 0
    object_seq_info = []
    object_seq_name_list = []
    for seq_name in tqdm(seq_name_list):
        # if verbose:
        #     print("-------------------")
        #     print(f"Processing sequence {seq_name}")
        seq_info = get_CO3D_seq_info(scene_info, scene_path, -1, seq_name)
        if not_filter:
            refined_seq_info = seq_info
        else:
            refined_seq_info = filter_seq_info(seq_info, iqr_factor, verbose)

        if len(refined_seq_info.seq_cameras) == 0:
            failed += 1
            continue

        object_seq_info.append(refined_seq_info)
        object_seq_name_list.append(seq_name)

    with open(object_seq_info_path, "wb") as f:
        pickle.dump(object_seq_info, f)

    object_seq_name_txt = "\n".join(object_seq_name_list)
    with open(object_seq_name_txt_path, "w") as f:
        f.write(object_seq_name_txt)        

    if verbose:
        print(len(object_seq_info), len(seq_name_list))
        print(f"failed: {failed}/{len(seq_name_list)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/mydata/data/hyunsoo/co3d_sample')
    parser.add_argument('--preprocess_dir', type=str, default='/mydata/data/hyunsoo/co3d_sample_preprocess')
    parser.add_argument('--not_filter', action='store_true')
    parser.add_argument('--iqr_factor', type=int, default=3)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    object_list = os.listdir(args.base_dir)
    object_list = [name for name in object_list if name not in ["co3d", "pytorch3d"]]
    object_list.sort()
    for object in object_list:
        if args.verbose:
            print("========================================")
            print(f"object: {object}")
        preprocess_CO3D_object(args.base_dir, args.preprocess_dir, object, args.iqr_factor, args.not_filter, args.verbose)