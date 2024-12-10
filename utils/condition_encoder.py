import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# LoRA를 적용한 Linear 레이어 정의
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=16, lora_alpha=1.0):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        # 기본 가중치 (동결 예정)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        if r > 0:
            # LoRA 가중치 (학습 대상)
            self.lora_A = nn.Parameter(torch.Tensor(r, in_features))
            self.lora_B = nn.Parameter(torch.Tensor(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.scaling = lora_alpha / r
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = None

    def forward(self, x):
        result = F.linear(x, self.weight)
        if self.r > 0:
            lora_result = F.linear(x, self.lora_A)
            lora_result = F.linear(lora_result, self.lora_B)
            result += self.scaling * lora_result
        return result

# LoRA를 적용한 Multihead Attention 정의
class LoRAMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, r=4, lora_alpha=1.0):
        super(LoRAMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "임베딩 차원은 헤드 수로 나누어 떨어져야 합니다."

        self.q_proj = LoRALinear(embed_dim, embed_dim, r=r, lora_alpha=lora_alpha)
        self.k_proj = LoRALinear(embed_dim, embed_dim, r=r, lora_alpha=lora_alpha)
        self.v_proj = LoRALinear(embed_dim, embed_dim, r=r, lora_alpha=lora_alpha)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        # query, key, value: (시퀀스 길이, 배치 크기, 임베딩 차원)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 멀티헤드 어텐션을 위한 형태 변환
        batch_size = query.size(1)
        q = q.view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # 어텐션 스코어 계산
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.head_dim)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_weights, v)

        # 원래 형태로 복원
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, batch_size, self.embed_dim)

        output = self.out_proj(attn_output)
        return output
    
class GuidanceModel(nn.Module):
    def __init__(
            self,
            input_dim=3,
            embed_dim=64,
            num_heads=4,
            output_dim=256,
            lora_r=16,
            lora_alpha=1.0,
            sample_rate=8,
            pool="None" # "None" or "Mean" or "Max"
            ):
        super(GuidanceModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.attention = LoRAMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, r=lora_r, lora_alpha=lora_alpha)
        # self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads).to(device)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.sample_rate = sample_rate
        self.pool = pool

    def sample(self, P):
        N = P.shape[0]
        downsampled_indices = np.random.choice(N, N // self.sample_rate, replace=False)
        P_tilde = P[downsampled_indices]
        return P_tilde

    def get_latent(self, P, P_tilde):
        P_embedded = self.embedding(P)
        P_tilde_embedded = self.embedding(P_tilde)

        P_embedded = P_embedded.unsqueeze(1)
        P_tilde_embedded = P_tilde_embedded.unsqueeze(1)

        latent = self.attention(
            query=P_tilde_embedded,  # 크기: (N/8, 1, embed_dim)
            key=P_embedded,          # 크기: (N, 1, embed_dim)
            value=P_embedded         # 크기: (N, 1, embed_dim)
        )  # 출력 크기: (N/8, 1, embed_dim)

        # 배치 차원 제거
        latent = latent.squeeze(1)  # 크기: (N/8, embed_dim)

        # 잠재 표현을 평균하여 고정 크기 벡터로 만듦
        if self.pool=="Mean":
            latent = latent.mean(dim=0)  # 크기: (embed_dim,)
        elif self.pool=="Max":
            latent = latent.max(dim=0).values # 크기: (embed_dim,)

        latent = self.ffn(latent)  # 최종 임베딩 (크기: (output_dim,))

        return latent
    
    def change_pool(self, pool):
        assert pool in ["None", "Mean", "Max"]
        self.pool = pool
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def save_lora(self, path):
        torch.save(self.attention.state_dict(), path)

    def load_lora(self, path):
        self.attention.load_state_dict(torch.load(path))

class SpatialGuidanceModel(GuidanceModel):
    def __init__(
            self,
            input_dim=3,
            embed_dim=64,
            num_heads=4,
            output_dim=768,
            lora_r=16,
            lora_alpha=1.0,
            sample_rate=8,
            pool="Mean"
            ):
        super(SpatialGuidanceModel, self).__init__(
            input_dim, embed_dim, num_heads, output_dim, lora_r, lora_alpha, sample_rate, pool
        )

    def forward(self, P):
        # P shape (N, 3) -> (1, N, 3) or (batch_size, N, 3)
        if len(P.shape) == 2:
            P = P.unsqueeze(0)
        
        # latent shape (batch_size, output_dim)
        latents = torch.cat([self.get_latent(P[i], self.sample(P[i])).unsqueeze(0) for i in range(P.shape[0])], dim=0)
        return latents

class TrajectoryGuidanceModel(GuidanceModel):
    def __init__(
            self,
            input_dim=8,
            embed_dim=64,
            num_heads=4,
            output_dim=768,
            lora_r=16,
            lora_alpha=1.0,
            sample_rate=2,
            pool="Mean"
            ):
        super(TrajectoryGuidanceModel, self).__init__(
            input_dim, embed_dim, num_heads, output_dim, lora_r, lora_alpha, sample_rate, pool
        )
    
    def add_time(self, T):
        N = T.shape[1]
        time = torch.linspace(0, 1, N).unsqueeze(0).repeat(T.shape[0], 1).unsqueeze(-1).to(T.device)
        T = torch.cat([T, time], dim=-1)
        return T

    def forward(self, T):
        # T shape (N, 7) -> (1, N, 7) or (batch_size, N, 7)
        if len(T.shape) == 2:
            T = T.unsqueeze(0)

        # T shape (batch_size, N, 7) -> (batch_size, N, 8)
        T = self.add_time(T)

        # latent shape (batch_size, output_dim)
        latents = torch.cat([self.get_latent(T[i], self.sample(T[i])).unsqueeze(0) for i in range(T.shape[0])], dim=0)

        return latents

def load_spatial_guidance_model(path):
    model = SpatialGuidanceModel()
    model.load_state_dict(torch.load(path))
    return model

def load_trajectory_guidance_model(path):
    model = TrajectoryGuidanceModel()
    model.load_state_dict(torch.load(path))
    return model

class DemoPointCloudDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=1024):
        super(DemoPointCloudDataset, self).__init__()
        self.num_samples = num_samples
        self.num_points = num_points

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random point cloud data
        P_3d = np.random.rand(self.num_points, 3).astype(np.float32)
        return P_3d
    
class DemoTrajectoryDataset(Dataset):
    def __init__(self, num_samples=1000, num_poses=64):
        super(DemoTrajectoryDataset, self).__init__()
        self.num_samples = num_samples
        self.num_poses = num_poses

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random point cloud data
        T_3d = np.random.rand(self.num_poses, 7).astype(np.float32)
        return T_3d

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dim = 768

    spatial_model = SpatialGuidanceModel(output_dim=out_dim).to(device)
    trajectory_model = TrajectoryGuidanceModel(output_dim=out_dim).to(device)
    spatial_model.save_model(f"./guidance/spatial_guidance_model_{out_dim}.pth")
    trajectory_model.save_model(f"./guidance/trajectory_guidance_model_{out_dim}.pth")
    breakpoint()

    # spatial_model = load_spatial_guidance_model(f"./guidance/spatial_guidance_model_{spatial_pool}.pth")
    # trajectory_model = load_trajectory_guidance_model(f"./guidance/trajectory_guidance_model_{trajectory_pool}.pth")
    # spatial_model.load_lora(f"./guidance/spatial_guidance_model_{spatial_pool}_lora.pth")
    # trajectory_model.load_lora(f"./guidance/trajectory_guidance_model_{trajectory_pool}_lora.pth")

    # --------------------------------------------------------------
    # simple inference
    # 포인트 클라우드 P (N, 3) 생성 또는 로드
    # P_N = 1024  # 예시로 N을 1024로 설정
    # P = np.random.rand(P_N, 3)  # 실제 데이터로 교체하세요

    # T_N = 64
    # T = np.random.rand(T_N, 7)  # 실제 데이터로 교체하세요

    # # 텐서로 변환
    # P_tensor = torch.from_numpy(P).float()       # 크기: (N, 3)
    # P_tensor = P_tensor.to(device)

    # T_tensor = torch.from_numpy(T).float()       # 크기: (N, 7)
    # T_tensor = T_tensor.to(device)

    # P_embedding = spatial_model(P_tensor)
    # print(P_embedding.shape)  # 출력 차원 확인
    # # print(P_embedding)

    # T_embedding = trajectory_model(T_tensor)
    # print(T_embedding.shape)  # 출력 차원 확인
    # # print(T_embedding)

    # --------------------------------------------------------------
    # lora train - batch
    # 기본 모델 파라미터 동결 (LoRA 파라미터만 학습)
    for name, param in spatial_model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    for name, param in trajectory_model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    # 학습 가능한 파라미터 수 확인
    total_params = sum(p.numel() for p in spatial_model.parameters()) + sum(p.numel() for p in trajectory_model.parameters())
    trainable_params = sum(p.numel() for p in spatial_model.parameters() if p.requires_grad) + sum(p.numel() for p in trajectory_model.parameters() if p.requires_grad)
    print(f'총 파라미터 수: {total_params}, 학습 가능한 파라미터 수: {trainable_params}')

    # 옵티마이저 및 손실 함수 정의
    spatial_parameters = filter(lambda p: p.requires_grad, spatial_model.parameters())
    trajectory_parameters = filter(lambda p: p.requires_grad, trajectory_model.parameters())
    optimizer = optim.Adam(list(spatial_parameters) + list(trajectory_parameters), lr=1e-3)
    criterion = nn.MSELoss()

    # 데이터셋 및 데이터로더 생성
    spatial_dataset = DemoPointCloudDataset()
    trajectory_dataset = DemoTrajectoryDataset()

    spatial_dataloader = DataLoader(spatial_dataset, batch_size=32, shuffle=True)
    trajectory_dataloader = DataLoader(trajectory_dataset, batch_size=32, shuffle=True)

    # 학습 루프
    num_epochs = 10
    for epoch in range(num_epochs):
        spatial_model.train()
        trajectory_model.train()

        for P, T in zip(spatial_dataloader, trajectory_dataloader):
            P = P.to(device)
            T = T.to(device)

            # 순전파
            P_output = spatial_model(P)
            T_output = trajectory_model(T)

            # 임의의 타깃 벡터 생성 (예시용)
            P_target = torch.randn(P_output.shape).to(device)
            T_target = torch.randn(T_output.shape).to(device)

            # 손실 계산
            loss = criterion(P_output, P_target) + criterion(T_output, T_target)

            # 역전파 및 옵티마이저 스텝
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'epoch [{epoch+1}/{num_epochs}], loss: {loss.item():.4f}')
        spatial_model.save_lora(f"./guidance/spatial_guidance_model_{out_dim}_lora.pth")
        trajectory_model.save_lora(f"./guidance/trajectory_guidance_model_{out_dim}_lora.pth")
        
    # --------------------------------------------------------------
    # full train - batch
    # lora 제외 모든 파라미터 학습
    # for param in spatial_model.parameters():
    #     if 'lora' in param.name:
    #         param.requires_grad = False
    #     param.requires_grad = True
        
    # for param in trajectory_model.parameters():
    #     if 'lora' in param.name:
    #         param.requires_grad = False
    #     param.requires_grad = True
        
    # # 학습 가능한 파라미터 수 확인
    # total_params = sum(p.numel() for p in spatial_model.parameters()) + sum(p.numel() for p in trajectory_model.parameters())
    # trainable_params = sum(p.numel() for p in spatial_model.parameters() if p.requires_grad) + sum(p.numel() for p in trajectory_model.parameters() if p.requires_grad)
    # print(f'총 파라미터 수: {total_params}, 학습 가능한 파라미터 수: {trainable_params}')

    # # 옵티마이저 및 손실 함수 정의
    # spatial_parameters = filter(lambda p: p.requires_grad, spatial_model.parameters())
    # trajectory_parameters = filter(lambda p: p.requires_grad, trajectory_model.parameters())
    # optimizer = optim.Adam(list(spatial_parameters) + list(trajectory_parameters), lr=1e-3)
    # criterion = nn.MSELoss()

    # # 데이터셋 및 데이터로더 생성
    # spatial_dataset = DemoPointCloudDataset()
    # trajectory_dataset = DemoTrajectoryDataset()

    # spatial_dataloader = DataLoader(spatial_dataset, batch_size=32, shuffle=True)
    # trajectory_dataloader = DataLoader(trajectory_dataset, batch_size=32, shuffle=True)

    # # 학습 루프
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     spatial_model.train()
    #     trajectory_model.train()

    #     for P, T in zip(spatial_dataloader, trajectory_dataloader):
    #         P = P.to(device)
    #         T = T.to(device)

    #         # 순전파
    #         P_output = spatial_model(P)
    #         T_output = trajectory_model(T)

    #         # 임의의 타깃 벡터 생성 (예시용)
    #         P_target = torch.randn(P_output.shape).to(device)
    #         T_target = torch.randn(T_output.shape).to(device)

    #         # 손실 계산
    #         loss = criterion(P_output, P_target) + criterion(T_output, T_target)

    #         # 역전파 및 옵티마이저 스텝
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     print(f'epoch [{epoch+1}/{num_epochs}], loss: {loss.item():.4f}')
    #     spatial_model.save_model(f"./guidance/spatial_guidance_model_trained_{out_dim}.pth")
    #     trajectory_model.save_model(f"./guidance/trajectory_guidance_model_trained_{out_dim}.pth")
        
    