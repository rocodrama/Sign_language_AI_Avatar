
import json, math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
from torch.utils.data import Dataset

def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_first_key(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            arr = np.array(d[k], dtype=float)
            return arr
    return None

def _to_TxJxC(arr: np.ndarray, J: int):
    if arr is None: return None
    T = arr.shape[0]
    C = 3 if arr.shape[1] == J*3 else 2 if arr.shape[1] == J*2 else None
    if C is None: return None
    return arr.reshape(T, J, C)

def _center_and_scale(x: np.ndarray):
    if x is None: return None
    x = x.copy()
    J = x.shape[1]
    ref = x[:, :min(J, 12), :].mean(axis=1, keepdims=True)  # torso-ish mean
    x = x - ref
    rng = x[:, :min(J, 12), 1].ptp(axis=1).mean() + 1e-6
    x = x / rng
    return x

class SignDataset(Dataset):
    def __init__(self, index_csv: Path, data_root: Path, config: Dict[str, Any] = None, return_text=True):
        self.df = pd.read_csv(index_csv, encoding="utf-8-sig")
        self.root = Path(data_root)
        self.cfg = config or {}
        self.return_text = return_text

        keys = self.cfg.get("keys", {})
        self.k_pose3d = keys.get("pose3d", ["pose_keypoints_3d"])
        self.k_handL3d = keys.get("handL3d", ["hand_left_keypoints_3d"])
        self.k_handR3d = keys.get("handR3d", ["hand_right_keypoints_3d"])
        self.k_face3d  = keys.get("face3d",  ["face_keypoints_3d"])

        self.k_pose2d = keys.get("pose2d", ["pose_keypoints_2d"])
        self.k_handL2d = keys.get("handL2d", ["hand_left_keypoints_2d"])
        self.k_handR2d = keys.get("handR2d", ["hand_right_keypoints_2d"])
        self.k_face2d  = keys.get("face2d",  ["face_keypoints_2d"])

    def __len__(self):
        return len(self.df)

    def _load_keypoints(self, kp_path: Path):
        d = _read_json(kp_path)
        body = _get_first_key(d, self.k_pose3d) or _get_first_key(d, self.k_pose2d)
        lh   = _get_first_key(d, self.k_handL3d) or _get_first_key(d, self.k_handL2d)
        rh   = _get_first_key(d, self.k_handR3d) or _get_first_key(d, self.k_handR2d)
        face = _get_first_key(d, self.k_face3d)  or _get_first_key(d, self.k_face2d)

        body = _to_TxJxC(body, 25) if body is not None else None
        lh   = _to_TxJxC(lh,   21) if lh   is not None else None
        rh   = _to_TxJxC(rh,   21) if rh   is not None else None
        face = _to_TxJxC(face, 70) if face is not None else None

        parts = [p for p in [body, lh, rh, face] if p is not None]
        if not parts:
            raise ValueError("No keypoint arrays present in JSON")
        T = min(p.shape[0] for p in parts)
        parts = [p[:T] for p in parts]
        x = np.concatenate(parts, axis=1)  # [T, J, C]
        x = _center_and_scale(x)
        return {"pose": x, "T": x.shape[0], "J": x.shape[1], "C": x.shape[2]}

    def _load_text(self, morph_path: Path):
        try:
            d = _read_json(morph_path)
        except Exception:
            return {"text": None, "gloss": None}
        text = d.get("text") or d.get("korean") or d.get("sentence") or d.get("문장")
        gloss = d.get("gloss") or d.get("글로스") or d.get("morpheme") or d.get("형태소")
        return {"text": text, "gloss": gloss}

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        kp_rel = row["keypoint_json"]
        morph_rel = row.get("morpheme_json", None)

        kp_path = self.root / kp_rel if isinstance(kp_rel, str) else Path(kp_rel)
        data = self._load_keypoints(kp_path)

        text_blob = {"text": None, "gloss": None}
        if isinstance(morph_rel, str) and len(morph_rel):
            morph_path = self.root / morph_rel
            text_blob = self._load_text(morph_path)

        item = {
            "pose": torch.from_numpy(data["pose"]).float(),
            "meta": {
                "stem": row.get("stem", None),
                "view": row.get("view", None),
                "bucket": row.get("bucket", None),
                "dataset_type": row.get("dataset_type", None),
                "T": data["T"],
                "J": data["J"],
                "C": data["C"],
            }
        }
        if self.return_text:
            item["text"] = text_blob["text"]
            item["gloss"] = text_blob["gloss"]
        return item
