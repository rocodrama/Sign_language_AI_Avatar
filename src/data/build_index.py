#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progress-enabled dataset indexer for NIA-style sign-language sets.
- Works with videos or JSON-only.
- Shows a progress bar and per-step logs when --progress / --verbose are enabled.

Usage
  python build_index_v4.py \
      --root /path/to/DATA_ROOT \
      --out  /path/to/index.csv \
      [--relative-to /path/to/DATA_ROOT] \
      [--seed /path/to/seed.csv] [--seed-key stem] \
      [--progress] [--verbose]

"""
import argparse, re, sys, os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # if tqdm not available, fall back silently

VIEWS = set(list("FUDRL"))

def norm_sep(p: Path) -> str:
    return str(p).replace("\\", "/")

def stem_no_view(stem: str) -> str:
    m = re.match(r"^(.*)_[FUDRL]$", stem, flags=re.IGNORECASE)
    return m.group(1) if m else stem

def infer_view_from_name(name: str) -> Optional[str]:
    m = re.search(r"_([FUDRL])(?:\.\w+)?$", name, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r"_([FUDRL])_", name, re.IGNORECASE)
    if m: return m.group(1).upper()
    return None

def infer_source_from_name(name: str) -> Optional[str]:
    m = re.search(r"(REAL|SYN|CROWD\d*)", name, re.IGNORECASE)
    return m.group(1).upper() if m else None

def infer_dataset_type_from_path(p: Path) -> Optional[str]:
    low = norm_sep(p).lower()
    if "/sen/" in low or low.endswith("/sen"): return "SEN"
    if "/word/" in low or low.endswith("/word"): return "WORD"
    if "/fs/" in low or low.endswith("/fs"): return "FS"
    return None

def infer_bucket_from_path(p: Path) -> Optional[str]:
    parts = norm_sep(p).split("/")
    for token in parts:
        t = token.upper()
        if t in {"REAL", "SYN"} or t.startswith("CROWD"):
            return t
    return None

def read_seed(seed_path: Optional[Path], verbose=False) -> Optional[pd.DataFrame]:
    if not seed_path: return None
    encs = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
    seps = [",", "\t", ";", "|"]
    last = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(seed_path, encoding=enc, sep=sep, engine="python")
                if df.shape[1] >= 1:
                    if verbose:
                        print(f"[SEED] Loaded {seed_path} enc={enc} sep={repr(sep)} shape={df.shape}")
                    return df
            except Exception as e:
                last = e
    raise last if last else RuntimeError("Failed to read seed CSV")

def scan_files(root: Path, progress=False, verbose=False):
    videos: Dict[str, List[Path]] = {}
    kp_jsons: Dict[str, List[Path]] = {}
    morph_jsons: Dict[str, List[Path]] = {}

    it = root.rglob("*")
    if progress and tqdm is not None:
        it = tqdm(it, desc="Scanning files", unit="file")

    count = 0
    for p in it:
        if not p.is_file():
            continue
        name = p.name
        low = name.lower()
        s = p.stem
        count += 1
        if low.endswith(".mp4"):
            videos.setdefault(s, []).append(p)
        elif low.endswith(".json"):
            parent = p.parent.name.lower()
            if "keypoint" in parent or "pose" in parent:
                kp_jsons.setdefault(s, []).append(p)
            elif "morpheme" in parent or "morph" in parent:
                morph_jsons.setdefault(s, []).append(p)
            else:
                if ("keypoint" in low) or ("pose" in low):
                    kp_jsons.setdefault(s, []).append(p)
                if ("morpheme" in low) or ("morph" in low):
                    morph_jsons.setdefault(s, []).append(p)
    if verbose:
        print(f"[SCAN] videos={sum(len(v) for v in videos.values())}, keypoints={sum(len(v) for v in kp_jsons.values())}, morphs={sum(len(v) for v in morph_jsons.values())}")
    return videos, kp_jsons, morph_jsons

def choose_first(paths: List[Path], prefer_keywords: Tuple[str, ...]) -> Optional[Path]:
    if not paths: return None
    for p in sorted(paths):
        parent = p.parent.name.lower()
        if any(k in parent for k in prefer_keywords):
            return p
    return sorted(paths)[0]

def build_index(root: Path, seed_df: Optional[pd.DataFrame], seed_key: Optional[str], relative_to: Optional[Path], progress=False, verbose=False) -> pd.DataFrame:
    videos, kp_jsons, morph_jsons = scan_files(root, progress=progress, verbose=verbose)

    stems = set(videos.keys()) | set(kp_jsons.keys()) | set(morph_jsons.keys())
    stems |= {stem_no_view(s) for s in list(stems)}
    stems = sorted(stems)

    rows = []
    iterator = stems
    if progress and tqdm is not None:
        iterator = tqdm(stems, desc="Matching stems", unit="stem")

    for stem in iterator:
        vpaths = videos.get(stem) or videos.get(stem_no_view(stem)) or []
        kp_paths = kp_jsons.get(stem) or kp_jsons.get(stem_no_view(stem)) or []
        morph_paths = morph_jsons.get(stem) or morph_jsons.get(stem_no_view(stem)) or []

        vpath = choose_first(vpaths, ("video",))
        kpath = choose_first(kp_paths, ("keypoint","pose"))
        mpath = choose_first(morph_paths, ("morpheme","morph"))

        vname = (vpath.name if vpath else (kp_paths[0].name if kp_paths else (mpath.name if mpath else f"{stem}.mp4")))

        def rel(p: Optional[Path]) -> Optional[str]:
            if p is None: return None
            p = p.resolve()
            if relative_to:
                return norm_sep(p.relative_to(relative_to.resolve()))
            return norm_sep(p)

        view = infer_view_from_name(vname) or infer_view_from_name(stem)
        source = infer_source_from_name(vname) or infer_source_from_name(stem)
        hint_path = vpath or (kp_paths[0] if kp_paths else (mpath if mpath else root))
        dataset_type = infer_dataset_type_from_path(hint_path)
        bucket = infer_bucket_from_path(hint_path)

        rows.append({
            "stem": stem,
            "dataset_type": dataset_type,
            "bucket": bucket,
            "source": source,
            "view": view,
            "video_path": rel(vpath) if vpath else None,
            "keypoint_json": rel(kpath) if kpath else None,
            "morpheme_json": rel(mpath) if mpath else None,
            "has_video": bool(vpath),
            "has_keypoint": bool(kpath),
            "has_morpheme": bool(mpath),
        })

    out_df = pd.DataFrame(rows)

    if seed_df is not None:
        if seed_key is None:
            for cand in ["stem", "video_stem", "basename", "video", "file", "name"]:
                if cand in seed_df.columns:
                    seed_key = cand
                    break
        if seed_key is not None and seed_key in seed_df.columns:
            if verbose:
                print(f"[SEED] Merging on seed key: {seed_key}")
            out_df = out_df.merge(seed_df, left_on="stem", right_on=seed_key, how="left")
        else:
            print(f"[WARN] Could not find a suitable join key in seed; skipping merge. (seed_key={seed_key})")

    return out_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Dataset root to scan")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path")
    ap.add_argument("--relative-to", type=Path, default=None, help="Make paths relative to this dir")
    ap.add_argument("--seed", type=Path, default=None, help="Optional seed CSV")
    ap.add_argument("--seed-key", type=str, default=None, help="Join key in seed; auto-detect if omitted")
    ap.add_argument("--progress", action="store_true", help="Show progress bars (requires tqdm)")
    ap.add_argument("--verbose", action="store_true", help="Print detailed logs")
    args = ap.parse_args()

    root = args.root.resolve()
    seed_df = read_seed(args.seed, verbose=args.verbose)

    df = build_index(root, seed_df, args.seed_key, args.relative_to, progress=args.progress, verbose=args.verbose)

    # Save main CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # Save missing report
    miss = df[(~df["has_keypoint"]) | (~df["has_morpheme"]) | (~df["has_video"])]
    if not miss.empty:
        miss_path = args.out.with_name(args.out.stem + "_missing.csv")
        miss.to_csv(miss_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Missing report: {miss_path} (rows={len(miss)})")

    # Summary
    total = len(df)
    both_json = len(df[(df["has_keypoint"]) & (df["has_morpheme"])])
    print(f"[SUMMARY] total={total}, with_both_json={both_json}, with_video={len(df[df['has_video']])}")
    print(f"[WROTE] {args.out}")

if __name__ == "__main__":
    main()
