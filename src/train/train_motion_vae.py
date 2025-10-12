
import argparse, os, math, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_sen import SignDataset
from utils import collate_and_resample, flatten_TJC
from ldm_stub import MotionVAE

def loss_vae(recon, x, mu, logvar, beta=1e-3):
    recon_loss = nn.MSELoss()(recon, x)
    # KL(N(mu, sigma)||N(0,1))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--t_target", type=int, default=60, help="Resample each clip to this #frames")
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1e-3, help="KL weight")
    ap.add_argument("--out_dir", type=Path, default=Path("checkpoints_vae"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ds_tr = SignDataset(Path(args.train), Path(args.data_root))
    ds_va = SignDataset(Path(args.val), Path(args.data_root))

    def collate(items):
        return collate_and_resample(items, args.t_target)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Peek a batch to get dims
    batch0 = next(iter(dl_tr))
    X0 = batch0["pose"]  # [B, T, J, C]
    B, T, J, C = X0.shape
    in_dim = T*J*C

    model = MotionVAE(in_dim=in_dim, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss = tr_recon = tr_kl = 0.0
        n_tr = 0
        for batch in dl_tr:
            x = batch["pose"].to(device)  # [B, T, J, C]
            x, _ = flatten_TJC(x)         # [B, D]
            recon, mu, logvar = model(x)
            loss, rec, kl = loss_vae(recon, x, mu, logvar, beta=args.beta)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x.size(0)
            tr_loss += loss.item()*bs; tr_recon += rec*bs; tr_kl += kl*bs; n_tr += bs

        model.eval()
        va_loss = va_recon = va_kl = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in dl_va:
                x = batch["pose"].to(device)
                x, _ = flatten_TJC(x)
                recon, mu, logvar = model(x)
                loss, rec, kl = loss_vae(recon, x, mu, logvar, beta=args.beta)
                bs = x.size(0)
                va_loss += loss.item()*bs; va_recon += rec*bs; va_kl += kl*bs; n_va += bs

        tr_loss/=n_tr; tr_recon/=n_tr; tr_kl/=n_tr
        va_loss/=n_va; va_recon/=n_va; va_kl/=n_va

        print(f"[E{epoch:03d}] train: loss={tr_loss:.4f} recon={tr_recon:.4f} kl={tr_kl:.4f} | "
              f"val: loss={va_loss:.4f} recon={va_recon:.4f} kl={va_kl:.4f}")

        # checkpoint
        ckpt = out / f"vae_e{epoch:03d}_val{va_loss:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "in_dim": in_dim,
            "latent_dim": args.latent_dim,
            "t_target": args.t_target,
            "joints": (J, C),
        }, ckpt)
        if va_loss < best_val:
            best_val = va_loss
            best = out / "vae_best.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "in_dim": in_dim,
                "latent_dim": args.latent_dim,
                "t_target": args.t_target,
                "joints": (J, C),
            }, best)
            print(f"[SAVE] best -> {best}")

if __name__ == "__main__":
    main()
