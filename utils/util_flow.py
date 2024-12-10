import numpy as np
import matplotlib.pyplot as plt

def create_camera_intrinsics(fx=800, fy=800, cx=640, cy=480):
    """
    카메라 내부 파라미터 행렬을 생성합니다.
    """
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    return K

def project_points(P, R, T, K):
    """
    3D 포인트를 2D 이미지 좌표로 투영합니다.
    
    Parameters:
    - P: (N, 3) 3D 포인트 클라우드
    - R: (3, 3) 회전 행렬
    - T: (3,) 이동 벡터
    - K: (3, 3) 카메라 내부 파라미터 행렬
    
    Returns:
    - pts_2d: (N, 2) 2D 이미지 좌표
    """
    # 카메라 좌표계로 변환
    P_cam = (R @ P.T).T + T  # (N, 3)
    
    # 투영 (핀홀 카메라 모델)
    P_proj = (K @ P_cam.T).T  # (N, 3)
    
    # 정규화
    pts_2d = P_proj[:, :2] / P_proj[:, 2, np.newaxis]
    
    return pts_2d

def compute_optical_flow(P, R_source, T_source, R_query, T_query, K):
    """
    소스 뷰에서 쿼리 뷰로의 Optical Flow를 계산합니다.
    
    Parameters:
    - P: (N, 3) 3D 포인트 클라우드
    - R_source: (3, 3) 소스 뷰의 회전 행렬
    - T_source: (3,) 소스 뷰의 이동 벡터
    - R_query: (3, 3) 쿼리 뷰의 회전 행렬
    - T_query: (3,) 쿼리 뷰의 이동 벡터
    - K: (3, 3) 카메라 내부 파라미터 행렬
    
    Returns:
    - flow: (N, 2) Optical Flow (dx, dy)
    """
    # 소스 뷰에서의 2D 투영
    pts_source = project_points(P, R_source, T_source, K)  # (N, 2)
    
    # 쿼리 뷰에서의 2D 투영
    pts_query = project_points(P, R_query, T_query, K)  # (N, 2)
    
    # Optical Flow 계산
    flow = pts_query - pts_source  # (N, 2)
    
    return flow

def main():
    # 카메라 내부 파라미터
    K = create_camera_intrinsics(fx=800, fy=800, cx=640, cy=480)
    
    # 소스 뷰 1의 회전(R1)과 이동(T1)
    R1 = np.eye(3)  # 정면
    T1 = np.array([0, 0, 0])  # 원점
    
    # 소스 뷰 2의 회전(R2)과 이동(T2)
    angle = np.deg2rad(10)  # 10도 회전
    R2 = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    T2 = np.array([1, 0, 0])  # x축으로 1만큼 이동
    
    # 쿼리 뷰의 회전(Rq)과 이동(Tq)
    angle_q = np.deg2rad(5)  # 5도 회전
    Rq = np.array([
        [np.cos(angle_q), 0, np.sin(angle_q)],
        [0, 1, 0],
        [-np.sin(angle_q), 0, np.cos(angle_q)]
    ])
    Tq = np.array([0.5, 0, 0])  # x축으로 0.5만큼 이동
    
    # 3D 포인트 클라우드 생성 (예: 정면에 있는 점들)
    num_points = 1000
    np.random.seed(42)
    X = np.random.uniform(-1, 1, num_points)
    Y = np.random.uniform(-1, 1, num_points)
    Z = np.random.uniform(4, 6, num_points)  # 카메라로부터의 거리
    P = np.vstack((X, Y, Z)).T  # (N, 3)
    
    # Optical Flow 계산 (소스 뷰 1과 2에서 각각)
    flow1 = compute_optical_flow(P, R1, T1, Rq, Tq, K)
    flow2 = compute_optical_flow(P, R2, T2, Rq, Tq, K)
    
    # 결과 시각화 (각각의 Optical Flow를 화살표로 표시)
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    
    # 첫 번째 소스 뷰의 Optical Flow 화살표 시각화
    axes[0].quiver(P[:, 0], P[:, 1], flow1[:, 0], flow1[:, 1],
                   angles='xy', scale_units='xy', scale=0.1, color='r', alpha=0.5, width=0.002)
    axes[0].set_title('Optical Flow from Source View 1 to Query View')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].invert_yaxis()

    # 두 번째 소스 뷰의 Optical Flow 화살표 시각화
    axes[1].quiver(P[:, 0], P[:, 1], flow2[:, 0], flow2[:, 1],
                   angles='xy', scale_units='xy', scale=0.1, color='b', alpha=0.5, width=0.002)
    axes[1].set_title('Optical Flow from Source View 2 to Query View')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].invert_yaxis()

    # 이미지 크기에 맞춰 grid 추가
    for ax in axes:
        ax.set_xlim(0, 640)
        ax.set_ylim(0, 480)
        ax.set_aspect('equal')
        ax.grid(True, which='both', linestyle=':', color='gray', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Optical Flow 출력 (일부)
    print("Optical Flow (dx, dy) 예시 from Source View 1:")
    print(flow1[:5])
    print("Optical Flow (dx, dy) 예시 from Source View 2:")
    print(flow2[:5])

if __name__ == "__main__":
    main()