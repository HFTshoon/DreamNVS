import torch
import torch.nn as nn

class CameraPoseEncoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=512, num_heads=4, num_layers=2):
        super(CameraPoseEncoder, self).__init__()
        
        # Local Feature Extractor (Multi-Scale 1D-CNN)
        self.local_cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),  # Local scale
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),  # Broader scale
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3),  # Even broader scale
            nn.ReLU()
        )
        
        # Global Feature Extractor (Transformer Encoder)
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=256),
            num_layers=num_layers
        )
        
        # Projection to fixed-size representation
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        """
        x: (batch_size, 7, n)
        """
        # Local feature extraction
        x = self.local_cnn(x)  # (batch_size, hidden_dim, n)
        
        # Permute for transformer compatibility
        x = x.permute(2, 0, 1)  # (n, batch_size, hidden_dim)
        
        # Global feature extraction
        x = self.global_transformer(x)  # (n, batch_size, hidden_dim)
        
        # Temporal pooling (mean pooling across the trajectory)
        x = x.mean(dim=0)  # (batch_size, hidden_dim)
        
        # Project to 512 dimensions
        x = self.fc(x)  # (batch_size, output_dim)
        
        return x

# Example usage
batch_size = 8
n = 20  # Sequence length
input_dim = 7

encoder = CameraPoseEncoder(input_dim=input_dim)
input_vector = torch.randn(batch_size, 7, n)  # (batch_size, 7, n)
output_vector = encoder(input_vector)
print(output_vector.shape) 