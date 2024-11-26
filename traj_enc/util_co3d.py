import os
import pickle
import gzip
from typing import List, Tuple, Optional, Dict, Any, Type, IO, cast

import numpy as np

from util_traj import focal2fov, rotmat2qvec
from util_data import CameraInfo, SceneInfo, SeqInfo
from co3d.dataset.data_types import FrameAnnotation, load_dataclass_jgzip

def read_CO3D_scene_info(path):
    dataset_path = os.path.dirname(path)
    category_frame_annotations = load_dataclass_jgzip(os.path.join(path, "frame_annotations.jgz"), List[FrameAnnotation])
    cam_infos = []
    for frame in category_frame_annotations:
        R = np.array(frame.viewpoint.R)
        T = np.array(frame.viewpoint.T)
        image_path = os.path.join(dataset_path,frame.image.path)
        image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)
        width = frame.image.size[1]
        height = frame.image.size[0]
        FovX = focal2fov(frame.viewpoint.focal_length[0] * width / 2, width)
        FovY = focal2fov(frame.viewpoint.focal_length[1] * height / 2, height)
        cam_info = CameraInfo(uid=1, R=R, T=T, FovY=FovY, FovX=FovX, # image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    cam_infos = sorted(cam_infos.copy(), key = lambda x : x.image_name)
    scene_info = SceneInfo(train_cameras=cam_infos, test_cameras=[])
    return scene_info
    
def load_CO3D_data(path):
    assert path.endswith("preprocess"), "Please provide the preprocess folder"

    object_list = os.listdir(path)
    object_list.sort()
    data = {}
    data["object_list"] = object_list
    seq_cnt = 0
    for object in object_list:
        object_data = load_CO3D_object_data(path, object)
        data[object] = object_data
        seq_cnt += len(object_data)
    print(f"Loaded {seq_cnt} sequences from {len(object_list)} objects")
    return data

def load_CO3D_object_data(path, object):
    scene_path = os.path.join(path, object)

    seq_name_list_txt_path = os.path.join(scene_path, "object_seq_name.txt")
    with open(seq_name_list_txt_path, "r") as f:
        seq_name_list = f.readlines()
    seq_name_list = list(map(str.strip, seq_name_list))
    seq_name_list.sort()

    object_seq_info_path = os.path.join(scene_path, "object_seq_info.pkl")
    with open(object_seq_info_path, "rb") as f:
        object_seq_info = pickle.load(f)

    object_data = {}
    object_data["object_name"] = object
    object_seq_name_list = []
    for seq_info in object_seq_info:
        if len(seq_info.seq_cameras) == 0:
            continue
        
        seq_name = seq_info.seq_cameras[0].image_path.split("/")[-3]
        assert seq_name in seq_name_list, f"seq_name {seq_name} not in seq_name_list"
        object_seq_name_list.append(seq_name)

        object_data[seq_name] = seq_info
    # print(f"Loaded {len(object_data)} sequences for object {object}")
    object_data["seq_name_list"] = object_seq_name_list
    return object_data

def load_CO3D_traj_data(path):
    assert path.endswith("preprocess"), "Please provide the preprocess folder"

    object_list = os.listdir(path)
    object_list.sort()
    data = []
    seq_cnt = 0
    for object in object_list:
        object_data = load_CO3D_object_traj_data(path, object)
        data += object_data
        seq_cnt += len(object_data)
    print(f"Loaded {seq_cnt} sequences from {len(object_list)} objects")
    return data

def load_CO3D_object_traj_data(path, object):
    scene_path = os.path.join(path, object)

    seq_name_list_txt_path = os.path.join(scene_path, "object_seq_name.txt")
    with open(seq_name_list_txt_path, "r") as f:
        seq_name_list = f.readlines()
    seq_name_list = list(map(str.strip, seq_name_list))
    seq_name_list.sort()

    object_seq_info_path = os.path.join(scene_path, "object_seq_info.pkl")
    with open(object_seq_info_path, "rb") as f:
        object_seq_info = pickle.load(f)

    object_data = [] 
    for seq_info in object_seq_info:
        if len(seq_info.seq_cameras) == 0:
            continue
        
        seq_name = seq_info.seq_cameras[0].image_path.split("/")[-3]
        assert seq_name in seq_name_list, f"seq_name {seq_name} not in seq_name_list"

        seq_data = []
        for cam_info in seq_info.seq_cameras:
            pose = np.eye(4)
            pose[:3, :3] = cam_info.R
            pose[:3, 3] = cam_info.T
            pose = np.linalg.inv(pose)

            qvec = rotmat2qvec(pose[:3, :3])
            tvec = pose[:3, 3]
            pose_data = np.concatenate([qvec, tvec])
            seq_data.append(pose_data)
        object_data.append((np.array(seq_data), seq_name))
    # print(f"Loaded {len(object_data)} sequences for object {object}")
    return object_data

def get_CO3D_seq_info(scene_info, base_dir, seq_num=0, seq_name=""):
    if seq_name == "":
        assert seq_num >= 0
        seq_name_list = os.listdir(base_dir)
        seq_name_list = [name for name in seq_name_list if name[0].isdigit()]
        seq_name_list.sort()
        seq_name = seq_name_list[seq_num]

    seq_cam_infos = []
    for camera_info in scene_info.train_cameras:
        if camera_info.image_path.startswith(os.path.join(base_dir, seq_name)):
            seq_cam_infos.append(camera_info)

    seq_cam_infos = sorted(seq_cam_infos.copy(), key = lambda x : x.image_name)

    assert len(seq_cam_infos) > 0

    average_distance = np.mean([np.linalg.norm(camera_info.T) for camera_info in seq_cam_infos]) 

    seq_info = SeqInfo(seq_cameras=seq_cam_infos, average_distance=average_distance)
    return seq_info
