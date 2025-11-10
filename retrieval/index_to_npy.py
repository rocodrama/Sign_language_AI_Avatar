#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_to_npy.py
- index.csv(train/val/test)를 읽어, 각 stem의 frame JSON들을 [T,J,C] .npy로 패킹
- NumPy 2.0 호환 (np.ptp 미사용)
- Python 3.8/3.9 호환 (Union '|' 미사용)
- 빈 프레임/토르소 0개/스케일 0 등 예외를 견고하게 처리
"""

import argparse, json, re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# 예: NIA_SL_WORD0001_REAL01_F_000000000123_keypoints.json
#
STEM_RE = re.compile(r"(.*)_(\d{6,})_keypoints\.json$", re.IGNORECASE)

def norm_sep(p: Path) -> str:
    """ 경로 구분자를 POSIX 스타일(/)로 정규화합니다. """
    return str(p).replace("\\", "/")

def nanptp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Range = nanmax - nanmin (NumPy 2.0-safe)."""
    return np.nanmax(a, axis=axis) - np.nanmin(a, axis=axis)

def load_frame_json(p: Path) -> Optional[np.ndarray]:
    """한 프레임 JSON을 [J,C]로 반환 (C=2 또는 3, score 제거). 실패 시 None."""
    try:
        d: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
        
    # 'people' 키가 리스트일 수도, 딕셔너리일 수도 있음
    ppl = d.get("people", d.get("People", {}))
    
    if isinstance(ppl, list) and len(ppl) > 0:
        ppl = ppl[0]
    elif not isinstance(ppl, dict):
        ppl = {} 

    def pick3d2d(k3: str, k2: str):
        v3 = ppl.get(k3, None)
        v2 = ppl.get(k2, None)
        
        if v3 is None and v2 is None:
            v3 = d.get(k3, None)
            v2 = d.get(k2, None)
            
        if v3 is not None and len(v3) > 0: return np.asarray(v3, dtype=float), 3
        if v2 is not None and len(v2) > 0: return np.asarray(v2, dtype=float), 2
        return None, 0

    body, bC = pick3d2d("pose_keypoints_3d", "pose_keypoints_2d")
    lh,   lC = pick3d2d("hand_left_keypoints_3d", "hand_left_keypoints_2d")
    rh,   rC = pick3d2d("hand_right_keypoints_3d", "hand_right_keypoints_2d")
    face, fC = pick3d2d("face_keypoints_3d", "face_keypoints_2d")

    parts: List[np.ndarray] = []

    def take_coords(vec: np.ndarray, C: int) -> Optional[np.ndarray]:
        if C not in (2, 3):
            return None
            
        # --- (사용자 오류 지점) ---
        # 'step'은 (x,y,score) 또는 (x,y,z,score) 처럼
        # '좌표 + 신뢰도' 1세트의 크기를 의미합니다.
        # C=2 (2D) -> step=3
        # C=3 (3D) -> step=4
        # 이 코드는 정상입니다. IDE가 'step'을 잘못 인식하는 것일 수 있습니다.
        step = C + 1  
        
        if vec.size % step == 0:
            # (x,y,score) 또는 (x,y,z,score) 형식의 데이터
            arr = vec.reshape(-1, step)[:, :C] # score (마지막 값) 제외
        elif vec.size % C == 0:
            # (x,y) 또는 (x,y,z) 형식 (score가 없는 경우)
            arr = vec.reshape(-1, C)
        else:
            return None
        return arr

    for vec, C_val in ((body, bC), (lh, lC), (rh, rC), (face, fC)):
        if vec is None or C_val == 0:
            continue
        arr = take_coords(vec, C_val)
        if arr is None:
            continue
        parts.append(arr)

    if not parts:
        return None

    coord_dim = parts[0].shape[1]
    if any(p.shape[1] != coord_dim for p in parts):
        coord_dim = 2
        parts = [p[:, :2] for p in parts]

    try:
        frame = np.concatenate(parts, axis=0)  # [J, C]
    except Exception:
        return None
    if frame.ndim != 2 or frame.shape[0] == 0 or frame.shape[1] not in (2, 3):
        return None
    return frame

def collect_frames(first_json: Path) -> Tuple[Optional[str], List[Path]]:
    """ 대표 keypoint_json 파일명으로 동일한 stem의 모든 프레임 json을 찾음 """
    m = STEM_RE.match(first_json.name)
    if not m:
        print(f"[Warning] Could not parse stem from {first_json.name}")
        return None, []
        
    stem = m.group(1)  # 공통 stem (예: 'NIA_SL_WORD0001_REAL01F')
    pattern = f"{stem}_" + "[0-9]"*12 + "_keypoints.json"
    frames = sorted(first_json.parent.glob(pattern))
    return stem, frames

def resolve_path(root: Path, val: str) -> Path:
    """ 
    CSV에 저장된 상대 경로를 data_root 기준으로 절대 경로화.
    (build_index.py (v4)가 data/keypoints/... 로 저장했기 때문)
    """
    p = Path(val)
    if p.is_absolute():
        return p
    return (root / val)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Input index.csv (e.g., train.csv)")
    ap.add_argument("--data_root", required=True, help="Absolute path to the data root (e.g., D:\\ub\\ubiplay)")
    ap.add_argument("--out_root", required=True, help="Output directory for packed .npy files (e.g., D:\\My_Portable_Project\\npy_data)")
    ap.add_argument("--relative-to", default=None, help="(DEPRECATED - now inferred from data_root and out_root)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")
    ap.add_argument("--progress", action="store_true", help="Show progress bar")
    args = ap.parse_args()

    df = pd.read_csv(args.index, encoding="utf-8-sig")
    root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    
    # "휴대용 프로젝트"의 루트 폴더 (out_root의 부모)
    # 예: D:\My_Portable_Project
    portable_root = out_root.parent

    npy_paths: List[Optional[str]] = []
    fail_reason: List[Optional[str]] = []

    iterator = df.iterrows()
    if args.progress and tqdm is not None:
        iterator = tqdm(iterator, total=len(df), desc="Packing")

    ok = 0
    for i, row in iterator:
        kp = row.get("keypoint_json")
        if not isinstance(kp, str) or len(kp) == 0:
            npy_paths.append(None); fail_reason.append("no_keypoint_path"); continue

        # CSV의 경로는 data_root (D:\ub\ubiplay) 기준임
        kp_path = resolve_path(root, kp)
        if not kp_path.exists():
            npy_paths.append(None); fail_reason.append(f"keypoint_path_missing: {kp}"); continue

        stem, frames = collect_frames(kp_path)
        if not frames:
            npy_paths.append(None); fail_reason.append("no_frames_found"); continue

        # npy 파일 저장 경로 (절대 경로)
        # 예: D:\My_Portable_Project\npy_data\NIA_...F.npy
        npy_file_path = out_root / f"{stem}.npy"
        
        # CSV에 저장될 "휴대용" 상대 경로
        # 예: npy_data/NIA_...F.npy
        try:
            npy_relative_path = npy_file_path.relative_to(portable_root)
        except ValueError:
            npy_relative_path = npy_file_path # 상대 경로화 실패 시 절대 경로
            
        npy_path_str = norm_sep(Path(npy_relative_path))


        if npy_file_path.exists() and not args.overwrite:
            npy_paths.append(npy_path_str); fail_reason.append(None); ok += 1; continue

        seq: List[np.ndarray] = []
        for fp in frames:
            arr = load_frame_json(fp)
            if arr is None:
                continue
            seq.append(arr[None, ...])  # [1,J,C]

        if not seq:
            npy_paths.append(None); fail_reason.append("all_frames_bad"); continue

        X = np.concatenate(seq, axis=0).astype("float32")  # [T,J,C]
        T, J, C = X.shape

        # --- Normalize: root-center + height scale ---
        torso_J = min(J, 12) # pose_keypoints의 상체 부분
        if torso_J <= 0:
            npy_paths.append(None); fail_reason.append("no_torso_joints"); continue

        torso = X[:, :torso_J, :]                         # [T, torso_J, C]
        
        with np.errstate(all="ignore"):
            center = np.nanmean(torso, axis=1, keepdims=True)  # [T,1,C]
        if not np.isfinite(center).any():
            npy_paths.append(None); fail_reason.append("center_nan"); continue

        X = X - np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)

        with np.errstate(all="ignore"):
            y_range_per_frame = nanptp(torso[..., 1], axis=1)  # [T]
            scale = float(np.nanmean(y_range_per_frame))
        if (not np.isfinite(scale)) or scale <= 0:
            scale = 1.0 
        X = X / scale

        try:
            np.save(npy_file_path, X)
            npy_paths.append(npy_path_str); fail_reason.append(None); ok += 1
        except Exception:
            npy_paths.append(None); fail_reason.append("save_failed")

    df["keypoint_npy"] = npy_paths
    df["pack_fail_reason"] = fail_reason
    
    # 원본 CSV와 같은 폴더에 _withnpy.csv로 저장
    out_index = Path(args.index).with_name(Path(args.index).stem + "_withnpy.csv")
    df.to_csv(out_index, index=False, encoding="utf-8-sig")

    # 실패 행만 따로 저장 (디버그용)
    fail_out_path = Path(args.index).with_name(Path(args.index).stem + "_pack_failures.csv")
    fail_df = df[df["keypoint_npy"].isna() | (df["keypoint_npy"] == "")]
    if len(fail_df) > 0:
        fail_df.to_csv(fail_out_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] packed={ok}/{len(df)} -> {out_index}")
    if len(fail_df) > 0:
        print(f"[INFO] failures={len(fail_df)} -> {fail_out_path}")

if __name__ == "__main__":
    main()
