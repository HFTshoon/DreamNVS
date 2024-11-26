import torch
import torch.nn as nn

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=512, encoding_dim=512, num_heads=8, num_layers=6, max_len=100):
        super(TransformerAutoencoder, self).__init__()
        # 인코더 설정
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, max_len + 1, hidden_dim))  # +1은 CLS 토큰 포함
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, encoding_dim)  # 512차원으로 압축

        # 디코더 설정
        self.decoder_fc = nn.Linear(encoding_dim, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, seq_len, mask=None):
        batch_size, _, _ = x.size()
        
        # Input embedding + CLS token + Position embedding
        x = self.embedding(x)
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)  # [CLS] + 시퀀스 결합
        x = x + self.position_embedding[:, :seq_len + 1]  # Position Embedding 적용

        # Transformer 인코딩
        x = x.permute(1, 0, 2)  # (seq_len+1, batch, hidden_dim)
        if mask is not None:
            mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device), mask), dim=1)
        encoded_output = self.encoder(x, src_key_padding_mask=mask)

        # CLS 토큰의 출력만 추출
        cls_output = encoded_output[0]  # (batch, hidden_dim)
        encoding = self.fc(cls_output)  # 512차원으로 변환

        return encoding

    def decode(self, encoding, seq_len, mask=None):
        batch_size = encoding.size(0)
        
        # 디코더 초기화
        encoding = self.decoder_fc(encoding).unsqueeze(0)  # (1, batch, hidden_dim)
        
        # 가짜 시퀀스 생성 (디코더에서 입력으로 필요)
        tgt = torch.zeros(seq_len, batch_size, encoding.size(-1), device=encoding.device)
        # tgt = encoding.repeat(seq_len, 1, 1)
        
        if mask is not None:
            tgt_mask = mask
            tgt_mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device), tgt_mask), dim=1)
            tgt_mask = tgt_mask[:, 1:]  # 첫 CLS 토큰 제외

        decoded_output = self.decoder(
            tgt, encoding, tgt_key_padding_mask=tgt_mask
        )

        # 디코딩된 시퀀스를 원래 차원으로 매핑
        decoded = self.output_layer(decoded_output.permute(1, 0, 2))  # (batch, seq_len, input_dim)

        return decoded

    def forward(self, x, seq_len, mask=None):
        encoding = self.encode(x, seq_len, mask)
        decoded = self.decode(encoding, seq_len, mask)
        return decoded
    
class VAEEncoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=512, latent_dim=512, num_layers=2):
        super(VAEEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of latent distribution
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log-variance of latent distribution

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # Extract hidden state from LSTM
        hidden = hidden[-1]  # Use the last layer's hidden state
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
    
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=512, output_dim=7, num_layers=2):
        super(VAEDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)  # Map latent vector to hidden state
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.fc_out = nn.Linear(hidden_dim, output_dim)  # Output layer

    def forward(self, z, seq_len):
        hidden = self.fc(z).unsqueeze(1).repeat(1, seq_len, 1)  # Expand latent vector across the sequence
        output, _ = self.lstm(hidden)
        reconstructed = self.fc_out(output)
        return reconstructed
    
class VAE(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=512, latent_dim=512, num_layers=2):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim, num_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Reparameterization trick

    def forward(self, x, seq_len):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z, seq_len)
        return reconstructed # mu, logvar
    
