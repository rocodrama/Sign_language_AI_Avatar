#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
NPY-only VAE training script.

- Uses CSV with 'keypoint_npy' column (absolute or data_root-relative).
- Resamples every sequence to --t_target frames.
- Select GPU with --gpu: 'auto' (default), '-1'/'cpu', integer index (e.g., 0), or 'cuda:1'.

Example (Windows):
  python src\train\train_motion_vae_npyonly.py ^
    --train data\dataset\train_withnpy.csv ^
    --val   data\dataset\val_withnpy.csv ^
    --data_root C:\Users\Bry\ub\ubiplay ^
    --epochs 30 --batch_size 32 --t_target 60 --latent_dim 256 ^
    --gpu 0 --num_workers 2 --out_dir runs\checkpoints_vae
"""

import argparse
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------- GPU picker ----------------
def pick_device(gpu_arg: str) -> torch.device:
    ga = str(gpu_arg).strip().lower()
    if ga in ("-1", "cpu", "none"):
        return torch.device("cpu")
    if ga == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # numeric index or 'cuda:1' form
    try:
        idx = int(ga)
        dev = torch.device(f"cuda:{idx}")
    except ValueError:
        dev = torch.device(gpu_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to CPU.")
        return torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    return dev


# ---------------- Minimal VAE ----------------
class MotionVAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU()
        )
        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, in_dim)
        )

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1e-3):
    recon_loss = nn.MSELoss()(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, float(recon_loss.item()), float(kl.item())


# ---------------- Utils: resample/pad/flatten ----------------
def resample_T(x: np.ndarray, t_out: int) -> np.ndarray:
    """x: [T,J,C] -> [t_out,J,C] using 1D linear interpolation per (J,C)."""
    T, J, C = x.shape
    if T == t_out:
        return x
    t_in = np.linspace(0.0, 1.0, T, dtype=np.float32)
    t_go = np.linspace(0.0, 1.0, t_out, dtype=np.float32)
    y = np.empty((t_out, J, C), dtype=x.dtype)
    for j in range(J):
        for c in range(C):
            y[:, j, c] = np.interp(t_go, t_in, x[:, j, c])
    return y


def pad_J(x: np.ndarray, J_max: int) -> np.ndarray:
    """Pad joints to J_max with zeros. x: [T,J,C] -> [T,J_max,C]."""
    T, J, C = x.shape
    if J == J_max:
        return x
    y = np.zeros((T, J_max, C), dtype=x.dtype)
    y[:, :J, :] = x
    return y


def flatten_TJC(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """x: [B,T,J,C] -> flat: [B, T*J*C]. Returns (flat, (T,J,C))."""
    B, T, J, C = x.shape
    return x.reshape(B, T * J * C), (T, J, C)


# ---------------- Dataset (NPY only) ----------------
class NpyOnlyDataset(Dataset):
    def __init__(self, csv_path: Path, data_root: Path):
        self.root = Path(data_root)
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "keypoint_npy" not in df.columns:
            raise RuntimeError("CSV에 keypoint_npy 컬럼이 없습니다. index_to_npy.py 결과 CSV를 사용하세요.")
        ok = df["keypoint_npy"].notna() & (df["keypoint_npy"].astype(str).str.len() > 0)
        self.df = df[ok].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = str(row["keypoint_npy"])
        p = Path(path) if Path(path).is_absolute() else (self.root / path)
        X = np.load(p)  # [T,J,C], float32
        return {"pose": torch.from_numpy(X).float()}


# ---------------- Top-level collate (picklable on Windows) ----------------
def collate_fn(items, t_target: int):
    # items: list of {"pose": FloatTensor [T,J,C]}
    arrs = [it["pose"].numpy() for it in items]  # CPU collate
    J_max = max(a.shape[1] for a in arrs)
    out = []
    for a in arrs:
        a2 = resample_T(a, t_target)
        a3 = pad_J(a2, J_max)
        out.append(a3)
    X = torch.from_numpy(np.stack(out, axis=0)).float()  # [B,T,Jmax,C]
    return {"pose": X}


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--t_target", type=int, default=60)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=Path, default=Path("checkpoints_vae"))
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--gpu", type=str, default="auto",
                    help="GPU select: 'auto' (default), '-1'/'cpu', index (e.g., 0), or 'cuda:1'")
    args = ap.parse_args()

    device = pick_device(args.gpu)
    print(f"[INFO] device = {device}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ds_tr = NpyOnlyDataset(Path(args.train), Path(args.data_root))
    ds_va = NpyOnlyDataset(Path(args.val), Path(args.data_root))

    collate = partial(collate_fn, t_target=args.t_target)
    pin_mem = (device.type == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       collate_fn=collate, num_workers=args.num_workers, pin_memory=pin_mem)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       collate_fn=collate, num_workers=args.num_workers, pin_memory=pin_mem)

    # Peek shapes
    X0 = next(iter(dl_tr))["pose"]
    B, T, J, C = X0.shape
    in_dim = T * J * C
    print(f"[INFO] first batch shape = {tuple(X0.shape)}  -> in_dim={in_dim}")

    model = MotionVAE(in_dim=in_dim, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = float("inf")
    bestp = out / "vae_best.pt"

    for e in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        totL = totR = totK = n = 0
        for batch in dl_tr:
            x = batch["pose"].to(device, non_blocking=True)   # [B,T,J,C]
            x_flat, _ = flatten_TJC(x)
            recon, mu, logvar = model(x_flat)
            loss, rec, kl = vae_loss(recon, x_flat, mu, logvar, beta=args.beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x_flat.size(0)
            totL += loss.item() * bs
            totR += rec * bs
            totK += kl * bs
            n += bs
        trL, trR, trK = totL / n, totR / n, totK / n

        # ---- Val ----
        model.eval()
        vL = vR = vK = m = 0
        with torch.no_grad():
            for batch in dl_va:
                x = batch["pose"].to(device, non_blocking=True)
                x_flat, _ = flatten_TJC(x)
                recon, mu, logvar = model(x_flat)
                loss, rec, kl = vae_loss(recon, x_flat, mu, logvar, beta=args.beta)
                bs = x_flat.size(0)
                vL += loss.item() * bs
                vR += rec * bs
                vK += kl * bs
                m += bs
        vL, vR, vK = vL / m, vR / m, vK / m

        print(f"[E{e:03d}] train: loss={trL:.4f} recon={trR:.4f} kl={trK:.4f} "
              f"| val: loss={vL:.4f} recon={vR:.4f} kl={vK:.4f}")

        # save each epoch + best
        ck = out / f"vae_e{e:03d}_val{vL:.4f}.pt"
        torch.save({"epoch": e, "state_dict": model.state_dict(),
                    "in_dim": in_dim, "latent_dim": args.latent_dim,
                    "t_target": T, "joints": (J, C)}, ck)
        if vL < best:
            best = vL
            torch.save({"epoch": e, "state_dict": model.state_dict(),
                        "in_dim": in_dim, "latent_dim": args.latent_dim,
                        "t_target": T, "joints": (J, C)}, bestp)
            print(f"[SAVE] best -> {bestp}")

    print("[DONE] training finished.")

if __name__ == "__main__":
    main()
