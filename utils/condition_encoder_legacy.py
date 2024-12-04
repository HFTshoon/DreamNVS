import numpy as np
import torch
import torch.nn as nn
    
class GuidanceModel(nn.Module):
    def __init__(
            self,
            num_freqs_3d=128,
            num_freqs_traj=48,
            nhead_3d=8,
            nhead_traj=8,
            dim_feedforward_3d=256,
            dim_feedforward_traj=256
            ):
        super(GuidanceModel, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pos_emb_3d = PositionalEmbedding(num_freqs_3d).to(device)
        self.pos_emb_traj = PositionalEmbedding(num_freqs_traj).to(device)

        embedding_dim_3d = 2 * 3 * num_freqs_3d
        embedding_dim_traj = 2 * 8 * num_freqs_traj

        self.encoder_3d = CrossAttentionEncoder(embedding_dim_3d, nhead_3d, dim_feedforward_3d).to(device)
        self.encoder_traj = CrossAttentionEncoder(embedding_dim_traj, nhead_traj, dim_feedforward_traj).to(device)


    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def compute_3D_guidance(self, P):
        """
        Computes the encoding F(P) for the input point cloud P.

        Parameters:
        - P: numpy array of shape (N, 3)
        - num_freqs: Number of frequency bands for positional encoding
        - nhead: Number of heads in multi-head attention
        - dim_feedforward: Dimension of the feed-forward network

        Returns:
        - F_P: Tensor of shape (N_tilde, embedding_dim)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert P to a PyTorch tensor
        P_tensor = torch.from_numpy(P).float().to(device)  # Shape: (N, 3)

        # Down-sample P to obtain ˜P
        N = P_tensor.shape[0]
        N_tilde = N // 8  # Down-sample to 1/8
        indices = np.random.choice(N, N_tilde, replace=False)
        P_tilde_tensor = P_tensor[indices]  # Shape: (N_tilde, 3)

        # Positional Embedding
        P_emb = self.pos_emb_3d(P_tensor)             # Shape: (N, embedding_dim)
        P_tilde_emb = self.pos_emb_3d(P_tilde_tensor) # Shape: (N_tilde, embedding_dim)

        # Cross-Attention Encoding Module
        F_P = self.encoder_3d(P_tilde_emb, P_emb)  # Shape: (N_tilde, embedding_dim)

        return F_P  # Tensor of shape (N_tilde, embedding_dim)

    def compute_trajectory_guidance(self, P):
        """
        Computes the encoding F(P) for the input point cloud P.

        Parameters:
        - P: numpy array of shape (N, 7)
        - num_freqs: Number of frequency bands for positional encoding
        - nhead: Number of heads in multi-head attention
        - dim_feedforward: Dimension of the feed-forward network

        Returns:
        - F_P: Tensor of shape (N_tilde, embedding_dim)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # multiply pi to the quaternion
        P[:, 4:] *= np.pi

        # add time dimension
        time = np.arange(P.shape[0])/(P.shape[0]-1)
        P = np.concatenate([P, time.reshape(-1, 1)], axis=1)

        # Convert P to a PyTorch tensor
        P_tensor = torch.from_numpy(P).float().to(device)  # Shape: (N, 8)

        # Down-sample P to obtain ˜P
        N = P_tensor.shape[0]
        N_tilde = N // 2  # Down-sample to 1/2
        indices = np.random.choice(N, N_tilde, replace=False)
        P_tilde_tensor = P_tensor[indices]  # Shape: (N_tilde, 3)

        # Positional Embedding
        P_emb = self.pos_emb_traj(P_tensor)             # Shape: (N, embedding_dim)
        P_tilde_emb = self.pos_emb_traj(P_tilde_tensor) # Shape: (N_tilde, embedding_dim)

        # Cross-Attention Encoding Module
        F_P = self.encoder_traj(P_tilde_emb, P_emb)  # Shape: (N_tilde, embedding_dim)

        return F_P  # Tensor of shape (N_tilde, embedding_dim)
    
    def __call__(self, P_3d=None, P_traj=None):
        return self.forward(P_3d, P_traj)

    def forward(self, P_3d=None, P_traj=None):
        """
        Computes the encoding F(P) for the input point cloud P.

        Parameters:
        - P_3d: numpy array of shape (N, 3)
        - P_traj: numpy array of shape (N, 7)

        Returns:
        - F_P_3d: Tensor of shape (N_tilde_3d, embedding_dim)
        - F_P_traj: Tensor of shape (N_tilde_traj, embedding_dim)
        """
        if P_3d is not None:
            F_P_3d = self.compute_3D_guidance(P_3d)
        else:
            F_P_3d = None
        
        if P_traj is not None:
            F_P_traj = self.compute_trajectory_guidance(P_traj)
        else:
            F_P_traj = None
        return F_P_3d, F_P_traj

class PositionalEmbedding(nn.Module):
    def __init__(self, num_freqs):
        super(PositionalEmbedding, self).__init__()
        self.num_freqs = num_freqs
        # Create frequency bands for positional encoding
        self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs) * np.pi
        self.freq_bands = self.freq_bands.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, coords):
        """
        coords: Tensor of shape (N, k)
        Returns: Tensor of shape (N, embedding_dim = 2 * k * num_freqs)
        """
        coords = coords.unsqueeze(-1)  # Shape: (N, k, 1)
        emb = coords * self.freq_bands  # Broadcasting to shape: (N, k, num_freqs)
        emb = emb.view(coords.shape[0], -1)  # Flatten to (N, k * num_freqs)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # Shape: (N, 2 * k * num_freqs)
        return emb

class CrossAttentionEncoder(nn.Module):
    def __init__(self, embedding_dim, nhead, dim_feedforward, final_dim=1):
        super(CrossAttentionEncoder, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embedding_dim, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, embedding_dim)
        )

    def forward(self, query, key):
        """
        query: Tensor of shape (N_tilde, embedding_dim)
        key: Tensor of shape (N, embedding_dim)
        Returns: Tensor of shape (N_tilde, embedding_dim)
        """
        # Add batch dimension (batch_size=1)
        query = query.unsqueeze(1)  # Shape: (N_tilde, 1, embedding_dim)
        key = key.unsqueeze(1)      # Shape: (N, 1, embedding_dim)
        value = key                 # In cross-attention, value is typically the same as key

        # Perform cross-attention
        attn_output, _ = self.cross_attn(query, key, value)  # Output shape: (N_tilde, 1, embedding_dim)

        # Remove batch dimension
        attn_output = attn_output.squeeze(1)  # Shape: (N_tilde, embedding_dim)

        # Apply Feed-Forward Network
        output = self.ffn(attn_output)  # Shape: (N_tilde, embedding_dim)
        return output

def load_guidance_model(path):
    model = GuidanceModel()
    model.load_state_dict(torch.load(path))
    return model

# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = GuidanceModel().to(device)
    encoder.save_model("guidance/guidance_model.pth")
    # encoder = load_guidance_model("guidance/guidance_model.pth")

    # Generate an example point cloud P with N points
    N = 1024
    P_3d = np.random.rand(N, 3)  # Random point cloud data
    F_P_3d = encoder.compute_3D_guidance(P_3d)
    print("F(P_3d) shape:", F_P_3d.shape)

    N = 64
    P_traj = np.random.rand(N, 7) # Random trajectory data
    F_P_traj = encoder.compute_trajectory_guidance(P_traj)
    print("F(P_traj) shape:", F_P_traj.shape)


