# dataloader_utils.py
from typing import List, Dict, Any
import torch

def collate_sign_text(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    batch: [{pose: [T,J,C], text: str, meta: {...}}, ...]
    """
    B = len(batch)
    # T, J, C 추출
    lengths = [b["pose"].shape[0] for b in batch]
    max_T = max(lengths)
    J = batch[0]["pose"].shape[1]
    C = batch[0]["pose"].shape[2]

    poses = torch.zeros(B, max_T, J, C, dtype=batch[0]["pose"].dtype)
    # mask: True = padding
    pose_mask = torch.ones(B, max_T, dtype=torch.bool)

    texts = []
    metas = []

    for i, b in enumerate(batch):
        x = b["pose"]
        T = x.shape[0]
        poses[i, :T] = x
        pose_mask[i, :T] = False  # 실제 데이터 위치는 False
        texts.append(b["text"])
        metas.append(b["meta"])

    return {
        "pose": poses,          # [B, T_max, J, C]
        "pose_mask": pose_mask, # [B, T_max], bool
        "texts": texts,
        "metas": metas,
    }
