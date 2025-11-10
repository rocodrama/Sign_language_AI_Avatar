# models.py
import torch
import torch.nn as nn

class PoseEncoder(nn.Module):
    def __init__(self, J: int, C: int, d_model: int = 256, d_out: int = 256,
                 n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.J = J
        self.C = C

        self.input_proj = nn.Linear(J * C, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=1024,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.fc = nn.Linear(d_model, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, pose: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        pose: [B, T, J, C]
        mask: [B, T] (True = pad)
        return: [B, d_out], L2-normalized
        """
        B, T, J, C = pose.shape
        x = pose.view(B, T, J * C)  # [B, T, J*C]
        x = self.input_proj(x)      # [B, T, d_model]

        x = self.encoder(x, src_key_padding_mask=mask)  # [B, T, d_model]

        if mask is not None:
            lengths = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B,1]
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            x = x.sum(dim=1) / lengths
        else:
            x = x.mean(dim=1)

        x = self.fc(x)
        x = self.norm(x)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return x
