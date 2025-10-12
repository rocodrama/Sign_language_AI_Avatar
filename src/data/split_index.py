
import argparse, pandas as pd, numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.index, encoding="utf-8-sig")
    rng = np.random.default_rng(args.seed)

    # Basic filters: keep rows that have both JSONs
    if "has_keypoint" in df.columns:
        df = df[df["has_keypoint"] == True]
    if "has_morpheme" in df.columns:
        df = df[df["has_morpheme"] == True]
    df = df.reset_index(drop=True)

    # Stratify by dataset_type/bucket/view if present
    for col in ["dataset_type","bucket","view"]:
        if col not in df.columns:
            df[col] = "NA"

    groups = df.groupby(["dataset_type","bucket","view"])
    train_rows, val_rows, test_rows = [], [], []
    for _, g in groups:
        n = len(g)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = int(n * args.test_ratio)
        n_val = int(n * args.val_ratio)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test+n_val]
        train_idx = idx[n_test+n_val:]
        train_rows.append(g.iloc[train_idx])
        val_rows.append(g.iloc[val_idx])
        test_rows.append(g.iloc[test_idx])
    train = pd.concat(train_rows).sample(frac=1.0, random_state=args.seed)
    val = pd.concat(val_rows).sample(frac=1.0, random_state=args.seed+1)
    test = pd.concat(test_rows).sample(frac=1.0, random_state=args.seed+2)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir/"train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(out_dir/"val.csv", index=False, encoding="utf-8-sig")
    test.to_csv(out_dir/"test.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote: {out_dir/'train.csv'} ({len(train)})")
    print(f"Wrote: {out_dir/'val.csv'} ({len(val)})")
    print(f"Wrote: {out_dir/'test.csv'} ({len(test)})")

if __name__ == "__main__":
    main()
