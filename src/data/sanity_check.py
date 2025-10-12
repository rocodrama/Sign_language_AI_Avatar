
import argparse, json
from pathlib import Path
from dataset_sen import SignDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--num", type=int, default=3)
    args = ap.parse_args()

    ds = SignDataset(index_csv=Path(args.index), data_root=Path(args.data_root))
    n = min(args.num, len(ds))
    for i in range(n):
        item = ds[i]
        x = item["pose"]
        meta = item["meta"]
        print(f"[{i}] shape={tuple(x.shape)} T={meta['T']} J={meta['J']} C={meta['C']} view={meta['view']} bucket={meta['bucket']} stem={meta['stem']}")
        print(f"    text={str(item.get('text'))[:60]} gloss={str(item.get('gloss'))[:60]}")

if __name__ == "__main__":
    main()
