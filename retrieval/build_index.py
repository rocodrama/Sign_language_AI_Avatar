#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_index.py (v4)

- 개별적인 keypoint_root와 morpheme_root 디렉토리 지원
- 'real/word' 및 'syn/word' 데이터만 포함하도록 필터링
- 'F' (정면) 각도 데이터만 포함하도록 필터링
- morpheme.json의 'data[0].attributes[0].name'에서 gloss 추출
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # optional

def norm_sep(p: Path) -> str:
    """ 경로 구분자를 POSIX 스타일(/)로 정규화합니다. """
    return str(p).replace("\\", "/")

# 'NIA_SL_WORD0001_REAL01F_000000000000_keypoints.json' -> 'NIA_SL_WORD0001_REAL01F'
# 'NIA_SL_WORD0001_REAL01F_morpheme.json' -> 'NIA_SL_WORD0001_REAL01F'
#
STEM_SUFFIX_RE = re.compile(r"(?:_\d{6,})?(?:_(keypoints|morpheme))$", re.IGNORECASE)

def normalize_stem(stem: str) -> str:
    """ 파일명에서 프레임 번호와 _keypoints/_morpheme 접미사를 제거하여 공통 stem을 반환합니다. """
    return STEM_SUFFIX_RE.sub("", stem)

# --- 메타데이터 추론 함수 ---

def infer_view(name: str):
    """ 'F'가 포함되어 있는지 확인 (필터링된 상태이므로 'F'가 기본) """
    if name.upper().endswith('F'):
        return "F"
    m = re.search(r"_([FUDRL])(?:_|$)", name, re.IGNORECASE)
    return m.group(1).upper() if m else None

def infer_dataset_type(name: str, path: Path):
    """ 'WORD'가 포함되어 있는지 확인 (필터링된 상태이므로 'WORD'가 기본) """
    low = name.lower()
    if "word" in low or "/word/" in norm_sep(path).lower():
        return "WORD"
    if "sen" in low or "/sen/" in norm_sep(path).lower():
        return "SEN"
    return None

def infer_bucket(name: str, path: Path):
    """ 'REAL' 또는 'SYN' 추론 """
    m = re.search(r"(REAL\d*|SYN|CROWD\d*)", name, re.IGNORECASE)
    if m: return m.group(1).upper()
    for token in norm_sep(path).split("/"):
        t = token.upper()
        if t.startswith(("REAL","SYN","CROWD")):
            return t
    return None

# --- JSON 파싱 함수 (v3 로직) ---

def _read_json(p: Path) -> Dict[str, Any]:
    """ .json 파일을 읽어 딕셔너리로 반환 """
    if not p or not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def extract_gloss(morph_path: Optional[Path]) -> Optional[str]:
    """ 
    morpheme.json에서 gloss(형태소)를 추출합니다.
    NIA_SL_SEN0001_REAL01_F_morpheme.json 파일 구조(data[0].attributes[0].name)에 대응합니다.
    """
    if morph_path is None:
        return None
    d = _read_json(morph_path)
    
    try:
        data_list = d.get("data")
        if isinstance(data_list, list) and len(data_list) > 0:
            attr_list = data_list[0].get("attributes")
            if isinstance(attr_list, list) and len(attr_list) > 0:
                gloss = attr_list[0].get("name") # e.g., "왼쪽"
                if isinstance(gloss, str) and len(gloss) > 0:
                    return gloss
    except Exception:
        pass

    # (폴백 로직)
    gloss = d.get("gloss") or d.get("글로스") or d.get("morpheme") or d.get("형태소")
    if isinstance(gloss, str) and len(gloss) > 0:
        return gloss
        
    return None # Gloss 추출 실패

# =================================================

def main():
    parser = argparse.ArgumentParser(description="Build index.csv from separate keypoint and morpheme directories")
    parser.add_argument("--keypoint_root", type=Path, required=True, help="Absolute path to keypoints directory (e.g., D:\\ub\\ubiplay\\data\\keypoints)")
    parser.add_argument("--morpheme_root", type=Path, required=True, help="Absolute path to morpheme directory (e.g., D:\\ub\\ubiplay\\data\\morpheme)")
    parser.add_argument("--out", type=Path, required=True, help="Output path for the master index.csv")
    parser.add_argument("--relative-to", type=Path, default=None, help="Common ancestor path to make output paths relative (e.g., D:\\ub\\ubiplay)")
    parser.add_argument("--progress", action="store_true", help="Show progress bars")
    args = parser.parse_args()

    keypoint_map = {}  # norm_stem -> path
    morpheme_map = {}  # norm_stem -> {path, gloss, full_path}
    
    # 처리할 필터 (요청사항)
    filters = ["real/word", "syn/word"]
    
    relto_path = args.relative_to.resolve() if args.relative_to else None

    # --- 1. Morpheme 스캔 ---
    m_root = args.morpheme_root.resolve()
    morpheme_files = []
    for filt in filters:
        morpheme_files.extend(m_root.glob(f"{filt}/**/*_morpheme.json"))
        
    iterable = morpheme_files
    if args.progress and tqdm is not None:
        iterable = tqdm(morpheme_files, desc="Scanning Morphemes")

    for p in iterable:
        stem = normalize_stem(p.stem)
        # 'F' (정면) 각도 필터
        if not stem.upper().endswith('F'):
            continue
            
        gloss = extract_gloss(p)
        if gloss:
            morpheme_map[stem] = {'path': p, 'gloss': gloss, 'full_path': p}
        elif args.progress:
            print(f"\n[Warning] No gloss found in: {p.name}")

    print(f"Found {len(morpheme_map)} valid morpheme files.")

    # --- 2. Keypoint 스캔 ---
    k_root = args.keypoint_root.resolve()
    keypoint_files = []
    for filt in filters:
        keypoint_files.extend(k_root.glob(f"{filt}/**/*_keypoints.json"))

    iterable = keypoint_files
    if args.progress and tqdm is not None:
        iterable = tqdm(keypoint_files, desc="Scanning Keypoints")

    for p in iterable:
        stem = normalize_stem(p.stem)
        # 'F' (정면) 각도 필터
        if not stem.upper().endswith('F'):
            continue
            
        # 각 stem 당 하나의 대표 keypoint 파일만 저장 (000000 프레임)
        # index_to_npy.py는 이 대표 경로를 기준으로 나머지 프레임을 찾음
        if stem not in keypoint_map:
            keypoint_map[stem] = {'path': p, 'full_path': p}
            
    print(f"Found {len(keypoint_map)} unique keypoint stems.")

    # --- 3. 데이터 병합 ---
    stems = sorted(set(keypoint_map.keys()) | set(morpheme_map.keys()))
    rows = []
    
    iterable_stems = stems
    if args.progress and tqdm is not None:
        iterable_stems = tqdm(stems, desc="Building Index")

    for s in iterable_stems:
        k_data = keypoint_map.get(s)
        m_data = morpheme_map.get(s)

        kpath = k_data['path'] if k_data else None
        mpath = m_data['path'] if m_data else None
        gloss = m_data['gloss'] if m_data else None

        # 경로를 상대/절대로 출력
        def rel(p: Optional[Path]):
            if p is None:
                return None
            p_resolved = p.resolve()
            if relto_path:
                try:
                    return norm_sep(p_resolved.relative_to(relto_path))
                except ValueError:
                    # 다른 드라이브 등 공통 상위 폴더가 아닌 경우 절대 경로 사용
                    return norm_sep(p_resolved)
            return norm_sep(p_resolved)

        # 메타데이터 추론을 위한 힌트
        path_hint = m_data['full_path'] if m_data else (k_data['full_path'] if k_data else Path(s))
        
        dataset_type = infer_dataset_type(s, path_hint)
        bucket = infer_bucket(s, path_hint)
        view = infer_view(s)
        
        has_keypoint = bool(kpath)
        has_morpheme = bool(mpath) and (gloss is not None)

        rows.append({
            "stem": s,
            "dataset_type": dataset_type,
            "bucket": bucket,
            "view": view,
            "gloss": gloss,  # <<< gloss 컬럼
            "video_path": None,
            "keypoint_json": rel(kpath), # 대표 keypoint json 경로
            "morpheme_json": rel(mpath),
            "has_video": False,
            "has_keypoint": has_keypoint,
            "has_morpheme": has_morpheme,
        })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # 누락 리포트
    miss = df[(~df["has_keypoint"]) | (~df["has_morpheme"])]
    if not miss.empty:
        miss_path = args.out.with_name(args.out.stem + "_missing.csv")
        miss.to_csv(miss_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] missing rows: {len(miss)} -> {miss_path.name}")

    print(f"[SUMMARY] total stems={len(df)} | both_json={(df['has_keypoint'] & df['has_morpheme']).sum()}")
    print(f"[WROTE] {args.out}")

if __name__ == "__main__":
    main()
