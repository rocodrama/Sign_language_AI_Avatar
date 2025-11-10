# dataset_retrieval.py
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from dataset_sen import SignDataset

class SignTextPoseDataset(Dataset):
    """
    SignDataset을 랩핑해서
    - pose: [T, J, C] (torch.float32)
    - text: str (우선 순위: text > gloss)
    만 뽑아서 리트리벌 학습에 맞게 정리
    """
    def __init__(self, index_csv: str, data_root: str):
        self.base = SignDataset(index_csv=Path(index_csv),
                                data_root=Path(data_root),
                                config=None,
                                return_text=True)

        # 텍스트가 아예 없는 행은 미리 걸러주는 게 안정적일 수 있음
        self.valid_indices: List[int] = []
        for i in range(len(self.base)):
            row = self.base.df.iloc[i]
            morph_rel = row.get("morpheme_json", "")
            if isinstance(morph_rel, str) and len(morph_rel) > 0:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx = self.valid_indices[idx]
        item = self.base[base_idx]
        pose = item["pose"]              # [T, J, C], torch.FloatTensor
        text = item.get("text") or item.get("gloss")
        meta = item.get("meta", {})
        if text is None:
            # 혹시 모를 예외: text/gloss 둘 다 None이면 빈 문자열로
            text = ""
        return {
            "pose": pose,
            "text": text,
            "meta": meta,
        }
