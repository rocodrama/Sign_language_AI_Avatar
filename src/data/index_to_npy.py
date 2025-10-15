#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pack NIA-style frame JSONs into a single [T,J,C] .npy per stem,
and write an index_withnpy.csv that adds 'keypoint_npy' column.

- Compatible with NumPy 2.0 (no .ptp usage).
- Supports relative paths (use --relative-to same as build_index.py).
"""

import argparse, re, json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

STEM_RE = re.compile(r"(.*)_(\d{6,})_keypoints\.json$", re.IGNORECASE)

def nanptp(a, axis=None):
    """NumPy 2.0-safe range = nanmax - nanmin."""
    return np.nanmax(a, axis=axis) - np.nanmin(a, axis=axis)

def load_frame_json(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    ppl = d.get("people", d.get("People", {}))

    def pick3d2d(k3, k2):
        if isinstance(ppl, dict):
            v3 = ppl.get(k3, None)
            v2 = ppl.get(k2, None)
        else:
            v3 = d.get(k3, None)
            v2 = d.get(k2, None)
        if v3 is not None: return np.asarray(v3, dtype=float), 3
        if v2 is not None: return np.asarray(v2, dtype=float), 2
        return None, 0

    body, bC = pick3d2d("pose_keypoints_3d", "pose_keypoints_2d")
    lh,   lC = pick3d2d("hand_left_keypoints_3d", "hand_left_keypoints_2d")
    rh,   rC = pick3d2d("hand_right_keypoints_3d", "hand_right_keypoints_2d")
    face, fC = pick3d2d("face_keypoints_3d", "face_keypoints_2d")

    parts = []
    if body is not None: parts.append(("body", body, bC))
    if lh   is not None: parts.append(("lh",   lh,   lC))
    if rh   is not None: parts.append(("rh",   rh,   rC))
    if face is not None: parts.append(("face", face, fC))
    if not parts:
        return None

    out = []
    coord_dim = None
    for _, vec, C in parts:
        # vec: flattened (x,y[,z],score) repeats OR just coords
        step = C + 1  # assume score present
        if vec.size % step == 0:
            arr = vec.reshape(-1, step)[:, :C]  # drop score
        else:
            if vec.size % C != 0:
                return None
            arr = vec.reshape(-1, C)

        if coord_dim is None:
            coord_dim = arr.shape[1]
        else:
            if arr.shape[1] != coord_dim:
                coord_dim = 2
                out = [o[:, :2] for o in out]
                arr = arr[:, :2]
        out.append(arr)

    frame = np.concatenate(out, axis=0)  # [J, C]
    return frame

def collect_frames(first_json: Path):
    m = STEM_RE.match(first_json.name)
    if not m:
        return None, []
    stem = m.group(1)  # e.g., NIA_SL_WORD0001_REAL01_F
    pattern = f"{stem}_" + "[0-9]"*12 + "_keypoints.json"
    frames = sorted(first_json.parent.glob(pattern))
    return stem, frames

def resolve_path(root: Path, val: str, relative_to: Path|None):
    p = Path(val)
    if p.is_absolute():
        return p
    # treat as relative to data root
    return (root / val)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", required=True, help="Where to save packed .npy files")
    ap.add_argument("--relative-to", default=None, help="Same base used when building index (usually equals data_root)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing npy")
    args = ap.parse_args()

    df = pd.read_csv(args.index, encoding="utf-8-sig")
    root = Path(args.data_root)
    relto = Path(args.relative_to).resolve() if args.relative_to else None
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    npy_paths = []
    iterator = df.iterrows()
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(df), desc="Packing")

    ok = 0
    for i, row in iterator:
        kp = row.get("keypoint_json")
        if not isinstance(kp, str) or len(kp) == 0:
            npy_paths.append(None)
            continue

        kp_path = resolve_path(root, kp, relto)
        if not kp_path.exists():
            npy_paths.append(None)
            continue

        stem, frames = collect_frames(kp_path)
        if not frames:
            npy_paths.append(None)
            continue

        npy_path = out_root / f"{stem}.npy"
        if npy_path.exists() and not args.overwrite:
            npy_paths.append(str(npy_path))
            ok += 1
            continue

        seq = []
        for fp in frames:
            arr = load_frame_json(fp)
            if arr is None:
                continue
            seq.append(arr[None, ...])  # [1,J,C]
        if not seq:
            npy_paths.append(None)
            continue

        X = np.concatenate(seq, axis=0).astype("float32")  # [T,J,C]

        # Normalize: root-center + height scale (NumPy 2.0-safe)
        torso = X[:, :min(X.shape[1], 12), :]
        center = torso.mean(axis=1, keepdims=True)
        X -= center
        # y-range mean as scale
        y_range_per_frame = nanptp(torso[..., 1], axis=1)  # [T]
        scale = float(np.nanmean(y_range_per_frame) + 1e-6)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        X /= scale

        np.save(npy_path, X)
        npy_paths.append(str(npy_path))
        ok += 1

    df["keypoint_npy"] = npy_paths
    out_index = Path(args.index).with_name(Path(args.index).stem + "_withnpy.csv")
    df.to_csv(out_index, index=False, encoding="utf-8-sig")
    print(f"[DONE] packed={ok}/{len(df)} -> {out_index}")

if __name__ == "__main__":
    main()
