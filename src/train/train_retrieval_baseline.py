
import argparse, os, json, math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_sen import SignDataset

try:
    from sentence_transformers import SentenceTransformer
    have_text_model = True
except Exception:
    have_text_model = False

def motion_embed(batch):
    x = batch["pose"]  # list of [T, J, C]
    maxT = max(t.shape[0] for t in x)
    X = []
    for t in x:
        T, J, C = t.shape
        if T < maxT:
            pad = t[-1:].repeat(maxT-T, 1, 1)
            t = torch.cat([t, pad], dim=0)
        X.append(t)
    X = torch.stack(X, dim=0)  # [B, T, J, C]
    X = X.mean(dim=1)          # [B, J, C]
    X = X.flatten(1)           # [B, J*C]
    X = torch.nn.functional.normalize(X, dim=1)
    return X

def collate(items):
    poses = [it["pose"] for it in items]
    texts = [ (it.get("text") or it.get("gloss") or "") for it in items ]
    metas = [ it["meta"] for it in items ]
    return {"pose": poses, "text": texts, "meta": metas}

def eval_retrieval(loader, txt_model):
    all_m = []
    all_t = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch in loader:
        M = motion_embed(batch).to(device)
        all_m.append(M)
        if txt_model:
            T = txt_model.encode(batch["text"], convert_to_tensor=True, normalize_embeddings=True).to(device)
        else:
            T = torch.randn_like(M)
            T = torch.nn.functional.normalize(T, dim=1)
        all_t.append(T)
    M = torch.cat(all_m, dim=0)
    T = torch.cat(all_t, dim=0)
    sim = T @ M.T
    correct = (sim.argmax(dim=1) == torch.arange(sim.size(0), device=sim.device)).float().mean().item()
    return {"R@1": correct}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    train = SignDataset(Path(args.train), Path(args.data_root))
    val = SignDataset(Path(args.val), Path(args.data_root))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    txt_model = None
    if have_text_model:
        txt_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

    for epoch in range(args.epochs):
        metrics = eval_retrieval(val_loader, txt_model)
        print(f"[Epoch {epoch}] R@1={metrics['R@1']:.4f} (text->motion)")

if __name__ == "__main__":
    main()
