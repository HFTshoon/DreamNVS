import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from traj_data import TrajectoryDataset
from DreamNVS.traj_enc.traj_enc_dec import TransformerAutoencoder, VAE
from util_co3d import load_CO3D_traj_data
data = load_CO3D_traj_data("/mydata/data/hyunsoo/co3d_preprocess")
# data = load_CO3D_traj_data("/mydata/data/hyunsoo/co3d_sample_preprocess")

max_len = 100
dataset = TrajectoryDataset(data, max_len)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 모델 초기화 및 학습 설정
model = VAE(input_dim=7, hidden_dim=512, latent_dim=512)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 루프
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    total_t_l2_loss = 0
    total_q_norm_loss = 0
    total_q_sim_loss = 0
    for batch, (x, seq_len) in enumerate(dataloader):
        optimizer.zero_grad()
        
        x = x.to(device)

        # 모델로 입력 시퀀스를 복원
        output = model(x, max_len)

        # make output of mask to zero
        # mask = torch.zeros_like(x)
        # for i in range(len(seq_len)):
        #     mask[i, :2*seq_len[i]] = 1
        # output = output * mask

        q_pred = output[:, :, :4]
        t_pred = output[:, :, 4:]
        q_target = x[:, :, :4]
        t_target = x[:, :, 4:]

        q_norm_pred = torch.norm(q_pred, dim=-1, keepdim=True)
        # q_norm_target = torch.norm(q_target, dim=-1, keepdim=True)
        q_norm_target = torch.ones_like(q_norm_pred)
        loss_q_norm = nn.MSELoss()(q_norm_pred, q_norm_target)

        q_pred_normalized = q_pred / (q_norm_pred + 1e-8)
        q_target_normalized = q_target / (q_norm_target + 1e-8)
        q_sim = 1.0 - torch.sum(q_pred_normalized * q_target_normalized, dim=-1, keepdim=True)**2
        for i in range(len(seq_len)):
            q_sim[i, 2*seq_len[i]:] = 0
        loss_q_sim = nn.MSELoss()(q_sim, torch.zeros_like(q_sim))

        loss_t_l2 = nn.MSELoss()(t_pred, t_target)

        loss = loss_q_norm + loss_q_sim + loss_t_l2
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_t_l2_loss += loss_t_l2.item()
        total_q_norm_loss += loss_q_norm.item()
        total_q_sim_loss += loss_q_sim.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_t_l2_loss = total_t_l2_loss / len(dataloader)
    avg_q_norm_loss = total_q_norm_loss / len(dataloader)
    avg_q_sim_loss = total_q_sim_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, t_l2_loss: {avg_t_l2_loss:.4f}, q_norm_loss: {avg_q_norm_loss:.4f}, q_sim_loss: {avg_q_sim_loss:.4f}')

    if epoch%10 == 0:
        txt_content = ""
        for xdata, outputdata in zip(x[0], output[0]):
            txt_content += f"{' '.join([f'{x:.3f}' for x in xdata])}\n"
            txt_content += f"{' '.join([f'{x:.3f}' for x in outputdata])}\n"
            txt_content += "-----------------\n"
        os.makedirs("results_vae", exist_ok=True)
        with open(f"results_vae/output_{epoch}.txt", "w") as f:
            f.write(txt_content)