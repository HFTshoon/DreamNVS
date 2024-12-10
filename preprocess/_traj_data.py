import numpy as np
import torch
from torch.utils.data import Dataset

from util_traj import make_relative

class TrajectoryDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx][0]
        seq_len = len(seq)
        
        actual_max_len = self.max_len//2

        q_norm = np.linalg.norm(seq[:, 4:], axis=-1, keepdims=True)
        seq[:, 4:] = seq[:, 4:] / np.mean(q_norm)

        # actual_seq_len = seq_len
        actual_seq_len = np.random.randint(min(seq_len, 10), min(seq_len, actual_max_len)+1)  # 랜덤한 길이로 시퀀스 자르기
        actual_seq_start = np.random.randint(0, seq_len - actual_seq_len + 1)
        seq = seq[actual_seq_start:actual_seq_start+actual_seq_len]

        seq = np.array(make_relative(seq))
        seq = torch.tensor(seq, dtype=torch.float32)

        seq_reverse = seq.flip(0).clone()
        seq = torch.cat((seq, seq_reverse), dim=0)

        # 패딩 추가 (7*n 벡터를 7*max_len으로 맞추기 위해 zero-padding 사용)
        if actual_seq_len * 2 < self.max_len:
            padding = torch.zeros(self.max_len - actual_seq_len * 2, 7)
            seq = torch.cat((seq, padding), dim=0)
        else:
            seq = seq[:self.max_len]
        
        return seq, actual_seq_len  # 시퀀스와 실제 길이를 반환