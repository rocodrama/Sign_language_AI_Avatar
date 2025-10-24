#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, sys, inspect, os
from pathlib import Path
import pandas as pd
import numpy as np
import torch

def make_npyonly_df(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "keypoint_npy" not in df.columns:
        raise RuntimeError("CSV에 keypoint_npy 컬럼이 없습니다. 먼저 index_to_npy.py로 패킹해 주세요.")
    ok = df["keypoint_npy"].notna() & (df["keypoint_npy"].astype(str).str.len() > 0)
    return df[ok].reset_index(drop=True)

class NpyOnlyDataset:
    def __init__(self, index_csv: Path, data_root: Path):
        self.root = Path(data_root)
        self.df = make_npyonly_df(index_csv)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p = str(row["keypoint_npy"])
        q = Path(p) if Path(p).is_absolute() else (self.root / p)
        X = np.load(q)  # [T,J,C]
        pose = torch.from_numpy(X).float()
        T,J,C = pose.shape
        meta = {
            "stem": row.get("stem"),
            "view": row.get("view"),
            "bucket": row.get("bucket"),
            "dataset_type": row.get("dataset_type"),
            "T": int(T), "J": int(J), "C": int(C),
            "keypoint_npy": str(q)
        }
        return {"pose": pose, "meta": meta, "text": None, "gloss": None}

def main():
    ap = argparse.ArgumentParser(description="Sanity check (prefers NPY; falls back if SignDataset lacks prefer_npy)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--module_dir", type=str, default=None, help="dataset_sen.py가 들어있는 폴더")
    ap.add_argument("--npy_only", action="store_true", help="NPY가 없는 행은 제외")
    ap.add_argument("--show_paths", action="store_true")
    args = ap.parse_args()

    index_path = Path(args.index)
    data_root = Path(args.data_root)

    # SignDataset import (옵션)
    if args.module_dir:
        sys.path.append(args.module_dir)
    SignDataset = None
    try:
        from dataset_sen import SignDataset as _SD
        SignDataset = _SD
    except Exception:
        pass  # 없으면 NPY 전용으로 진행

    # NPY 집계 정보 출력
    try:
        df_all = pd.read_csv(index_path, encoding="utf-8-sig")
        if "keypoint_npy" in df_all.columns:
            ok = df_all["keypoint_npy"].notna() & (df_all["keypoint_npy"].astype(str).str.len()>0)
            print(f"[INFO] rows={len(df_all)}  npy_filled={ok.sum()}  empty_npy={(~ok).sum()}")
    except Exception:
        pass

    # 데이터셋 구성
    ds = None
    if SignDataset is not None:
        sig = inspect.signature(SignDataset.__init__)
        if "prefer_npy" in sig.parameters:
            # 신버전: npy를 우선 사용
            ds = SignDataset(index_csv=index_path, data_root=data_root,
                             prefer_npy=True, normalize=True, return_text=True)
        else:
            # 구버전: 인자 없이 시도
            try:
                ds = SignDataset(index_csv=index_path, data_root=data_root)
            except TypeError:
                ds = None

    # NPY 전용 폴백 (권장)
    if ds is None or args.npy_only:
        ds = NpyOnlyDataset(index_csv=index_path, data_root=data_root)
        print(f"[INFO] using NpyOnlyDataset with {len(ds)} items")

    n = min(args.num, len(ds))
    if n == 0:
        print("[WARN] 데이터가 0개입니다. CSV/경로를 확인하세요.")
        return

    for i in range(n):
        try:
            item = ds[i]
        except Exception as e:
            print(f"[{i}] LOAD-ERROR: {e}")
            continue

        pose = item.get("pose")
        meta = item.get("meta", {}) or {}
        shape = tuple(pose.shape) if pose is not None else None

        path_info = ""
        if args.show_paths:
            if meta.get("keypoint_npy"):
                path_info += f" npy={meta['keypoint_npy']}"

        print(f"[{i}] shape={shape} "
              f"T={meta.get('T')} J={meta.get('J')} C={meta.get('C')} "
              f"view={meta.get('view')} bucket={meta.get('bucket')} stem={meta.get('stem')}{path_info}")

if __name__ == "__main__":
    main()
