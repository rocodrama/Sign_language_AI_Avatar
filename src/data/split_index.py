
import argparse, pandas as pd, numpy as np, sys
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # optional

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Input index.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory for train/val/test CSVs")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify_cols", type=str, default="dataset_type,bucket,view",
                    help="Comma-separated column names to stratify by (default: dataset_type,bucket,view)")
    ap.add_argument("--progress", action="store_true", help="Show progress bars (requires tqdm)")
    ap.add_argument("--verbose", action="store_true", help="Print detailed logs")
    ap.add_argument("--min_group", type=int, default=5, help="Skip groups smaller than this size (kept in train)")
    args = ap.parse_args()

    df = pd.read_csv(args.index, encoding="utf-8-sig")
    n0 = len(df)
    if args.verbose:
        print(f"[LOAD] rows={n0} file={args.index}")

    # Basic filters: keep rows that have both JSONs
    if "has_keypoint" in df.columns:
        df = df[df["has_keypoint"] == True]
    if "has_morpheme" in df.columns:
        df = df[df["has_morpheme"] == True]
    df = df.reset_index(drop=True)
    if args.verbose:
        print(f"[FILTER] kept rows={len(df)} (removed {n0-len(df)})")

    # Make sure stratify columns exist
    strat_cols = [c.strip() for c in args.stratify_cols.split(",") if c.strip()]
    for col in strat_cols:
        if col not in df.columns:
            df[col] = "NA"

    # Group and split
    rng = np.random.default_rng(args.seed)
    groups = df.groupby(strat_cols, dropna=False)
    train_rows, val_rows, test_rows = [], [], []

    iterable = groups
    if args.progress and tqdm is not None:
        iterable = tqdm(groups, total=len(groups), desc="Stratified groups")

    stats = []  # for report
    for key, g in iterable:
        n = len(g)
        if isinstance(key, tuple):
            key_str = "/".join(str(k) for k in key)
        else:
            key_str = str(key)

        if n < args.min_group:
            # too small; keep all in train to avoid empty/imbalanced splits
            train_rows.append(g)
            stats.append({"group": key_str, "n": n, "train": n, "val": 0, "test": 0, "note": "small_group"})
            if args.verbose:
                print(f"[GROUP] {key_str:30s} n={n:4d} -> train(all)")
            continue

        idx = np.arange(n); rng.shuffle(idx)
        n_test = int(n * args.test_ratio)
        n_val = int(n * args.val_ratio)
        test_idx = idx[:n_test]
        val_idx  = idx[n_test:n_test+n_val]
        train_idx= idx[n_test+n_val:]

        train_rows.append(g.iloc[train_idx])
        val_rows.append(g.iloc[val_idx])
        test_rows.append(g.iloc[test_idx])

        stats.append({"group": key_str, "n": n, "train": len(train_idx), "val": len(val_idx), "test": len(test_idx), "note": ""})
        if args.verbose:
            print(f"[GROUP] {key_str:30s} n={n:4d} -> train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train = pd.concat(train_rows).sample(frac=1.0, random_state=args.seed).reset_index(drop=True) if train_rows else pd.DataFrame(columns=df.columns)
    val   = pd.concat(val_rows).sample(frac=1.0, random_state=args.seed+1).reset_index(drop=True) if val_rows else pd.DataFrame(columns=df.columns)
    test  = pd.concat(test_rows).sample(frac=1.0, random_state=args.seed+2).reset_index(drop=True) if test_rows else pd.DataFrame(columns=df.columns)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir/"train.csv"
    val_path   = out_dir/"val.csv"
    test_path  = out_dir/"test.csv"
    report_path= out_dir/"split_report.csv"

    train.to_csv(train_path, index=False, encoding="utf-8-sig")
    val.to_csv(val_path, index=False, encoding="utf-8-sig")
    test.to_csv(test_path, index=False, encoding="utf-8-sig")

    # Report
    rep = pd.DataFrame(stats)
    rep.to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"[WRITE] {train_path} ({len(train)})")
    print(f"[WRITE] {val_path} ({len(val)})")
    print(f"[WRITE] {test_path} ({len(test)})")
    print(f"[REPORT] {report_path} (groups={len(rep)})")

    # Class balance overview
    def dist(df_):
        if df_.empty: return {}
        return df_.groupby(strat_cols).size().to_dict()

    print("[DIST] train:", dist(train))
    print("[DIST] val  :", dist(val))
    print("[DIST] test :", dist(test))

if __name__ == "__main__":
    main()
