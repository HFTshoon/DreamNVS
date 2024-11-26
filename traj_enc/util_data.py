from typing import NamedTuple
import numpy as np


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    # image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list

class SeqInfo(NamedTuple):
    seq_cameras: list
    average_distance: float