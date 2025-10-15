#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re, sys
from pathlib import Path
import pandas as pd

def norm_sep(p: Path) -> str:
    return str(p).replace("\\", "/")

# 예: NIA_SL_WORD0001_REAL02_F_000000000001_keypoints  -> NIA_SL_WORD0001_REAL02_F
#     NIA_SL_WORD0001_REAL02_F_morpheme                -> NIA_SL_WORD0001_REAL02_F
STEM_SUFFIX_RE = re.compile(r"(?:_\d{6,})?(?:_(keypoints|morpheme))$", re.IGNORECASE)

def normalize_stem(stem: str) -> str:
    return STEM_SUFFIX_RE.sub("", stem)

def infer_view(name: str):
    m = re.search(r"_([FUDRL])(?:_|$)", name, re.IGNORECASE)
    return m.group(1).upper() if m else None

def infer_dataset_type(name: str, path: Path):
    low = name.lower()
    if "word" in low or "/word/" in norm_sep(path).lower():
        return "WORD"
    if "sen" in low or "/sen/" in norm_sep(path).lower():
        return "SEN"
    if "fs" in low or "/fs/" in norm_sep(path).lower():
        return "FS"
    return None

def infer_bucket(name: str, path: Path):
    # REAL02, SYN, CROWD 등의 토큰 힌트
    m = re.search(r"(REAL\d*|SYN|CROWD\d*)", name, re.IGNORECASE)
    if m: return m.group(1).upper()
    # 경로에서도 한 번 더
    for token in norm_sep(path).split("/"):
        t = token.upper()
        if t.startswith(("REAL","SYN","CROWD")):
            return t
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--relative-to", type=Path, default=None)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()

    keypoint_map = {}  # norm_stem -> path
    morph_map    = {}  # norm_stem -> path

    it = root.rglob("*.json")
    if args.progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc="Scanning JSONs")
        except Exception:
            pass

    for p in it:
        name = p.name
        stem = p.stem
        low  = name.lower()
        norm = normalize_stem(stem)
        if low.endswith("_keypoints.json"):
            keypoint_map.setdefault(norm, p)
        elif low.endswith("_morpheme.json"):
            morph_map.setdefault(norm, p)

    # 유니온으로 모든 스템을 모아 테이블 생성
    stems = sorted(set(keypoint_map.keys()) | set(morph_map.keys()))
    rows = []
    for s in stems:
        kpath = keypoint_map.get(s)
        mpath = morph_map.get(s)

        # 경로를 상대/절대로 출력
        def rel(p: Path|None):
            if p is None: return None
            p = p.resolve()
            if args.relative_to:
                return norm_sep(p.relative_to(args.relative_to.resolve()))
            return norm_sep(p)

        name_hint = (kpath or mpath).name if (kpath or mpath) else s
        dataset_type = infer_dataset_type(name_hint, (kpath or mpath or root))
        bucket = infer_bucket(name_hint, (kpath or mpath or root))
        view = infer_view(name_hint)

        rows.append({
            "stem": s,
            "dataset_type": dataset_type,
            "bucket": bucket,
            "view": view,
            "video_path": None,
            "keypoint_json": rel(kpath),
            "morpheme_json": rel(mpath),
            "has_video": False,
            "has_keypoint": bool(kpath),
            "has_morpheme": bool(mpath),
        })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # 누락 리포트
    miss = df[(~df["has_keypoint"]) | (~df["has_morpheme"])]
    if not miss.empty:
        miss.to_csv(args.out.with_name(args.out.stem + "_missing.csv"),
                    index=False, encoding="utf-8-sig")
        print(f"[INFO] missing rows: {len(miss)} -> {args.out.stem}_missing.csv")

    print(f"[SUMMARY] total={len(df)} both_json={(df['has_keypoint'] & df['has_morpheme']).sum()}")
    print(f"[WROTE] {args.out}")

if __name__ == "__main__":
    main()
