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

from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "klue/roberta-base", d_out: int = 256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.fc = nn.Linear(hidden_size, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [B, hidden]
        x = self.fc(cls)
        x = self.norm(x)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return x  # [B, d_out]

import torch
import torch.nn.functional as F

class SignTextRetrievalModel(nn.Module):
    def __init__(self, J, C, d_model=256, d_out=256,
                 text_model_name="klue/roberta-base"):
        super().__init__()
        self.pose_encoder = PoseEncoder(J=J, C=C,
                                        d_model=d_model,
                                        d_out=d_out)
        self.text_encoder = TextEncoder(model_name=text_model_name,
                                        d_out=d_out)

    def forward(self, pose, pose_mask, input_ids, attention_mask):
        p_emb = self.pose_encoder(pose, mask=pose_mask)              # [B,d]
        t_emb = self.text_encoder(input_ids, attention_mask)         # [B,d]
        return p_emb, t_emb

def clip_contrastive_loss(p_emb, t_emb, temperature: float = 0.07):
    """
    p_emb, t_emb: [B, d], 이미 L2-normalized라고 가정
    """
    logits = p_emb @ t_emb.t() / temperature  # [B, B]
    labels = torch.arange(p_emb.size(0), device=p_emb.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) * 0.5

