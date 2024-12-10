import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer

def generate_smooth_camera_path(R1, T1, R2, T2, N):
    """
    두 카메라의 회전 행렬과 위치 벡터가 주어졌을 때,
    카메라 레이 사이의 가장 가까운 점의 중점을 구의 중심으로 설정하고,
    구의 단면 원주를 따라 부드럽게 이동하는 카메라 경로를 생성합니다.

    Parameters:
    - R1, T1: 첫 번째 카메라의 회전 행렬 (3x3)과 위치 벡터 (3D 벡터)
    - R2, T2: 두 번째 카메라의 회전 행렬 (3x3)과 위치 벡터 (3D 벡터)
    - N: 스텝 수 (양의 정수)

    Returns:
    - positions: 카메라 위치 리스트 (N+1 개의 3D 벡터)
    - rotations: 회전 행렬 리스트 (N+1 개의 3x3 행렬)
    """
    # 카메라 레이 방향 벡터 (Z축)
    D1 = R1[:, 2]
    D2 = R2[:, 2]

    # 두 레이 사이의 가장 가까운 점 계산
    T1 = np.array(T1, dtype=float)
    T2 = np.array(T2, dtype=float)
    D1 = D1 / np.linalg.norm(D1)
    D2 = D2 / np.linalg.norm(D2)
    T21 = T2 - T1

    a = np.dot(D1, D1)
    b = np.dot(D1, D2)
    c = np.dot(D2, D2)
    d = np.dot(D1, T21)
    e = np.dot(D2, T21)

    denom = a * c - b * b
    if np.abs(denom) < 1e-6:
        print("Warning: Camera rays are parallel.")
        # 레이가 평행한 경우
        s = 0
        t = d / c
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom

    P1_closest = T1 + s * D1
    P2_closest = T2 + t * D2

    # 구의 중심 (두 가장 가까운 점의 중점)
    C = (P1_closest + P2_closest) / 2

    # 각 카메라 위치에서의 깊이 (구의 중심으로부터의 거리)
    depth1 = np.linalg.norm(T1 - C)
    depth2 = np.linalg.norm(T2 - C)

    # 카메라 위치를 반지름 1인 구에 투영
    T1_dir = T1 - C
    T2_dir = T2 - C
    T1_dir_norm = T1_dir / np.linalg.norm(T1_dir)
    T2_dir_norm = T2_dir / np.linalg.norm(T2_dir)
    T1_proj = C + T1_dir_norm
    T2_proj = C + T2_dir_norm

    # 두 투영된 위치 벡터 사이의 각도 계산
    dot = np.dot(T1_dir_norm, T2_dir_norm)
    dot = np.clip(dot, -1.0, 1.0)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)

    # 평면의 법선 벡터 계산 (구의 단면 원주 평면)
    plane_normal = np.cross(T1_dir_norm, T2_dir_norm)
    if np.linalg.norm(plane_normal) < 1e-6:
        # 두 벡터가 평행한 경우 임의의 업 벡터 사용
        plane_normal = np.array([0, 1, 0])
    else:
        plane_normal /= np.linalg.norm(plane_normal)

    positions = []
    rotations = []

    for i in range(N + 1):
        t = i / N
        # 깊이를 큐빅 보간으로 부드럽게 변화
        t_cubic = 3 * t ** 2 - 2 * t ** 3
        depth = depth1 + (depth2 - depth1) * t_cubic

        # 구면 선형 보간 (SLERP)으로 방향 벡터 보간
        if sin_omega < 1e-6:
            dir_interp = T1_dir_norm
        else:
            a = np.sin((1 - t) * omega) / sin_omega
            b = np.sin(t * omega) / sin_omega
            dir_interp = a * T1_dir_norm + b * T2_dir_norm
            dir_interp /= np.linalg.norm(dir_interp)

        # 현재 위치 계산
        position = C + dir_interp * depth
        positions.append(position)

        # 회전 행렬 계산 (카메라가 해당 방향을 바라보도록)
        z_axis = dir_interp
        x_axis = np.cross(plane_normal, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            # z축과 평면 법선 벡터가 평행한 경우
            x_axis = np.array([1, 0, 0])
        else:
            x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        R = np.column_stack((x_axis, y_axis, z_axis))
        rotations.append(R)

    return positions, rotations

if __name__ == "__main__":
    # 사용 예시
    # R1 = np.eye(3)
    # T1 = [0, 0, 0]
    # R2 = np.array([
    #     [0.6, 0, -0.8],
    #     [0, 1, 0],
    #     [0.8, 0, 0.6]
    # ])
    # T2 = [1, 0, 0]
    # N = 10

    R1 = np.array([
        [0.82418, -0.035466, -0.56522],
        [0.053797, 0.99843, 0.015795],
        [0.56377, -0.043425, 0.82479]
    ])
    T1 = [0.16302, 0.0064381, 0.054947]
    # scale 10
    T1 = [x * 10 for x in T1]
    R2 = np.array([
        [0.99997, -0.0063155, 0.0053283],
        [0.0063151, 0.99998, 0.000092527],
        [-0.0053288, -0.000058875, 0.99999]
    ])

    T2 = [0, 0, 0]
    N = 10
    positions, rotations = generate_smooth_camera_path(R1, T1, R2, T2, N)

    # 결과 출력
    visualizer = CameraPoseVisualizer([-3, 3], [-3, 3], [0, 3])
    h = 0
    for i in range(len(positions)):
        print(f"Step {i}:")
        print(f"Position: {positions[i]}")
        print(f"Rotation matrix:\n{rotations[i]}\n")
        
        R = np.array(rotations[i])
        T = np.array(positions[i]).reshape(-1,1)
        # R_i = R.T
        # T_i = -np.matmul(R.T, T)
        matrix = np.concatenate((np.concatenate((R, T), axis=1), np.array([[0,0,0,1]])), axis=0)
        visualizer.extrinsic2pyramid(matrix, plt.cm.rainbow(h / len(positions)), focal_len_scaled=1)
        h += 1
    visualizer.show()