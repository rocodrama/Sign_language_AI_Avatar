#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal dataset indexer for NIA-style sign-language sets.

Works in THREE scenarios:
  (A) Full set: videos + keypoint/morpheme JSONs
  (B) JSON-only: ONLY keypoint/morpheme JSONs present (no videos)  <-- your case
  (C) Seed-driven: merge extra columns from a seed CSV (train/val/manifest)

Outputs:
  - index.csv: rows keyed by "stem" (filename without extension), with:
      dataset_type(SEN/WORD/FS), bucket(REAL/SYN/CROWDxx), source, view(F/U/D/R/L),
      video_path (may be empty), keypoint_json, morpheme_json,
      has_video, has_keypoint, has_morpheme
  - index_missing.csv: subset where at least one of the three is missing

Matching rules:
  - Primary key is filename stem. If an exact stem match fails, the script
    tries again after stripping a trailing view suffix _F/_U/_D/_R/_L.
  - If multiple JSONs/videos match the same stem, prefer folders named
    "keypoint"/"pose" and "morpheme"/"morph", and pick the first in sorted order.

Usage example:
  python build_index_v3.py \
      --root /path/to/DATA_ROOT \
      --out  /path/to/index.csv \
      --relative-to /path/to/DATA_ROOT \
      [--seed /path/to/seed.csv] \
      [--seed-key stem]   # column name in seed to join on (default: auto)

"""

import argparse, re
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional

VIEWS = set(list("FUDRL"))

def norm_sep(p: Path) -> str:
    return str(p).replace("\\", "/")

def stem_no_view(stem: str) -> str:
    # remove trailing _F/_U/_D/_R/_L
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
    # e.g., REAL / SYN / CROWD under the first levels
    parts = norm_sep(p).split("/")
    for token in parts:
        t = token.upper()
        if t in {"REAL", "SYN"} or t.startswith("CROWD"):
            return t
    return None

def read_seed(seed_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not seed_path: return None
    encs = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
    seps = [",", "\t", ";", "|"]
    last = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(seed_path, encoding=enc, sep=sep, engine="python")
                if df.shape[1] >= 1:
                    return df
            except Exception as e:
                last = e
    raise last if last else RuntimeError("Failed to read seed CSV")

def scan_files(root: Path):
    videos: Dict[str, List[Path]] = {}
    kp_jsons: Dict[str, List[Path]] = {}
    morph_jsons: Dict[str, List[Path]] = {}

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        low = name.lower()
        s = p.stem
        if low.endswith(".mp4"):
            videos.setdefault(s, []).append(p)
        elif low.endswith(".json"):
            parent = p.parent.name.lower()
            if "keypoint" in parent or "pose" in parent:
                kp_jsons.setdefault(s, []).append(p)
            elif "morpheme" in parent or "morph" in parent:
                morph_jsons.setdefault(s, []).append(p)
            else:
                # fallback by filename keywords
                if ("keypoint" in low) or ("pose" in low):
                    kp_jsons.setdefault(s, []).append(p)
                if ("morpheme" in low) or ("morph" in low):
                    morph_jsons.setdefault(s, []).append(p)
    return videos, kp_jsons, morph_jsons

def choose_first(paths: List[Path], prefer_keywords: Tuple[str, ...]) -> Optional[Path]:
    if not paths: return None
    # Prefer directory keywords if available
    for p in sorted(paths):
        parent = p.parent.name.lower()
        if any(k in parent for k in prefer_keywords):
            return p
    return sorted(paths)[0]

def build_index(root: Path, seed_df: Optional[pd.DataFrame], seed_key: Optional[str], relative_to: Optional[Path]) -> pd.DataFrame:
    videos, kp_jsons, morph_jsons = scan_files(root)

    # Build the driving key set from union of stems (videos ∪ JSONs)
    stems = set(videos.keys()) | set(kp_jsons.keys()) | set(morph_jsons.keys())
    # Also add stems with view suffix stripped, to ensure coverage
    stems |= {stem_no_view(s) for s in list(stems)}

    rows = []
    for stem in sorted(stems):
        # try exact then no-view
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
        dataset_type = infer_dataset_type_from_path(vpath or (kp_paths[0] if kp_paths else (mpath if mpath else root)))
        bucket = infer_bucket_from_path(vpath or (kp_paths[0] if kp_paths else (mpath if mpath else root)))

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

    # Merge seed if given
    if seed_df is not None:
        # auto-detect join key in seed if not provided
        if seed_key is None:
            for cand in ["stem", "video_stem", "basename", "video", "file", "name"]:
                if cand in seed_df.columns:
                    seed_key = cand
                    break
        if seed_key is not None and seed_key in seed_df.columns:
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
    ap.add_argument("--seed-key", type=str, default=None, help="Column in seed to join on; defaults to auto-detect")
    args = ap.parse_args()

    root = args.root.resolve()
    seed_df = read_seed(args.seed)

    df = build_index(root, seed_df, args.seed_key, args.relative_to)

    # Save main CSV
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
