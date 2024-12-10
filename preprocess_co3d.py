import os
import pickle
import argparse

import numpy as np
from tqdm import tqdm

from preprocess.util_co3d import read_CO3D_scene_info, get_CO3D_pointcloud

def preprocess_CO3D_object(base_dir, preprocess_dir, object, sample_n=4096, verbose=False):
    scene_path = os.path.join(base_dir, object)
    scene_info = read_CO3D_scene_info(scene_path)

    preprocess_path = os.path.join(preprocess_dir, object)
    if not os.path.exists(preprocess_path):
        os.makedirs(preprocess_path)

    seq_name_list = os.listdir(scene_path)
    seq_name_list = [name for name in seq_name_list if os.path.isdir(os.path.join(scene_path, name))]
    seq_name_list.sort()

    for seq_name in tqdm(seq_name_list):
        #if os.path.exists(os.path.join(preprocess_path, seq_name)):
        #    continue

        if verbose:
            print("-------------------")
            print(f"Processing sequence {seq_name}")

        get_CO3D_pointcloud(scene_info, scene_path, preprocess_path, seq_name, sample_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/mydata/data/hyunsoo/co3d_sample')
    parser.add_argument('--preprocess_dir', type=str, default='/mydata/data/hyunsoo/co3d_sample_preprocess')
    parser.add_argument('--sample_n', type=int, default=4096)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    object_list = os.listdir(args.base_dir)
    object_list = [name for name in object_list if name not in ["co3d", "pytorch3d"]]
    object_list.sort()
    for object in object_list:
        if args.verbose:
            print("========================================")
            print(f"object: {object}")
        preprocess_CO3D_object(args.base_dir, args.preprocess_dir, object, args.sample_n, args.verbose)