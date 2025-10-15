#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

STEM_RE = re.compile(r"(.*)_(\d{6,})_keypoints\.json$", re.IGNORECASE)

def load_frame_json(p: Path):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    ppl = d.get("people", {})
    # 3D 우선, 없으면 2D
    def pick3d2d(k3, k2):
        if k3 in ppl and ppl[k3] is not None: return ppl[k3], 3
        if k2 in ppl and ppl[k2] is not None: return ppl[k2], 2
        return None, 0

    body, bC = pick3d2d("pose_keypoints_3d", "pose_keypoints_2d")
    lh,   lC = pick3d2d("hand_left_keypoints_3d", "hand_left_keypoints_2d")
    rh,   rC = pick3d2d("hand_right_keypoints_3d", "hand_right_keypoints_2d")
    face, fC = pick3d2d("face_keypoints_3d", "face_keypoints_2d")

    parts = []
    if body is not None: parts.append(("body", np.array(body, dtype=float), bC))
    if lh   is not None: parts.append(("lh",   np.array(lh,   dtype=float), lC))
    if rh   is not None: parts.append(("rh",   np.array(rh,   dtype=float), rC))
    if face is not None: parts.append(("face", np.array(face, dtype=float), fC))
    if not parts:
        return None

    # 각 파트는 (x,y, z?, score) 반복 → 좌표만 추출
    out_vecs = []
    for name, arr, C in parts:
        step = C+1  # score 포함 길이
        # 일부 데이터는 C만 있을 수도 있어 방어
        has_score = (arr.size % step == 0)
        if has_score:
            arr = arr.reshape(-1, step)[:, :C]  # score 제거
        else:
            arr = arr.reshape(-1, C)
        out_vecs.append(arr)  # [J_part, C]
    xyz = np.concatenate(out_vecs, axis=0)  # [J, C]
    return xyz  # 1프레임

def collect_frames(first_json: Path):
    m = STEM_RE.match(first_json.name)
    if not m:
        return None, None
    stem = m.group(1)  # 공통 stem (예: NIA_SL_WORD0001_REAL01_F)
    # 동일 폴더의 같은 stem + 번호들
    pattern = f"{stem}_" + "[0-9]"*12 + "_keypoints.json"
    frames = sorted(first_json.parent.glob(pattern))
    return stem, frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", required=True, help="npy를 저장할 베이스 폴더")
    ap.add_argument("--relative-to", default=None, help="index 경로가 상대경로일 때 기준 루트(보통 data_root와 동일)")
    args = ap.parse_args()

    df = pd.read_csv(args.index, encoding="utf-8-sig")
    root = Path(args.data_root)
    relto = Path(args.relative_to).resolve() if args.relative_to else None
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    npy_paths = []
    ok = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Packing"):
        kp = row.get("keypoint_json")
        if not isinstance(kp, str) or len(kp)==0:
            npy_paths.append(None); continue
        kp_path = (root / kp) if (relto is not None or not Path(kp).is_absolute()) else Path(kp)
        stem, frames = collect_frames(kp_path)
        if not frames:
            npy_paths.append(None); continue

        seq = []
        for fp in frames:
            arr = load_frame_json(fp)
            if arr is None:
                continue
            seq.append(arr[None, ...])  # [1, J, C]
        if not seq:
            npy_paths.append(None); continue

        X = np.concatenate(seq, axis=0)  # [T, J, C]
        # 루트-센터/신장 스케일 간단 정규화
        torso = X[:, :min(X.shape[1],12), :]
        center = torso.mean(axis=1, keepdims=True)
        X = X - center
        scale = (torso[...,1].ptp(axis=1).mean() + 1e-6)
        X = X / scale

        npy_path = out_root / f"{stem}.npy"
        np.save(npy_path, X.astype("float32"))
        npy_paths.append(str(npy_path))
        ok += 1

    df["keypoint_npy"] = npy_paths
    out_index = Path(args.index).with_name(Path(args.index).stem + "_withnpy.csv")
    df.to_csv(out_index, index=False, encoding="utf-8-sig")
    print(f"[DONE] packed={ok}/{len(df)} → {out_index}")

if __name__ == "__main__":
    main()
