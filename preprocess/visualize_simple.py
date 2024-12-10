from typing import List
import matplotlib.pyplot as plt
from co3d.dataset.data_types import (load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation)
from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
import numpy as np
category_frame_annotations = load_dataclass_jgzip("frame_annotations.jgz", List[FrameAnnotation])
first_seq = category_frame_annotations[0:99]
visualizer = CameraPoseVisualizer([-10, 10], [0, 20], [-20, 0])
h = 0
for frame in first_seq:
    R = np.array(frame.viewpoint.R)
    T = np.array([frame.viewpoint.T]).T
    R_i = R.T
    T_i = -np.matmul(R.T, T)
    matrix = np.concatenate((np.concatenate((R_i, T_i), axis=1), np.array([[0,0,0,1]])), axis=0)
    visualizer.extrinsic2pyramid(matrix, plt.cm.rainbow(h / len(first_seq)), focal_len_scaled=1)
    h += 1
visualizer.show()