
import argparse, os, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from dataset_sen import SignDataset
from utils import collate_and_resample, flatten_TJC
from ldm_stub import MotionVAE, CondUNet1D

# Optional text encoder
try:
    from sentence_transformers import SentenceTransformer
    HAVE_TXT = True
except Exception:
    HAVE_TXT = False

def timesteps_schedule(T=1000):
    return torch.linspace(1e-4, 0.02, T)

class SimpleDDPM:
    def __init__(self, model, latent_dim, timesteps=1000, device="cpu"):
        self.model = model
        self.latent_dim = latent_dim
        self.T = timesteps
        self.device = device
        betas = timesteps_schedule(timesteps).to(device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def training_loss(self, x0, context):
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device)  # [B]
        alpha_bar_t = self.alpha_bar[t].view(B, 1)
        eps = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
        eps_pred = self.model(x_t, t_embed=None, context=context)
        loss = nn.MSELoss()(eps_pred, eps)
        return loss

def encode_text(texts):
    if HAVE_TXT:
        enc = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
        z = enc.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return z
    else:
        # Fallback: random but fixed-size context
        return torch.zeros((len(texts), 512))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--vae_ckpt", required=True, help="Path to trained VAE checkpoint (vae_best.pt)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--t_target", type=int, default=None, help="Overrides VAE t_target if set")
    ap.add_argument("--latent_dim", type=int, default=None, help="Overrides VAE latent if set")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out_dir", type=Path, default=Path("checkpoints_ldm"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Load VAE (frozen) to get encoder/decoder
    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    in_dim = ckpt["in_dim"]
    vae_latent = ckpt["latent_dim"]
    t_target = ckpt["t_target"]
    if args.t_target: t_target = args.t_target
    if args.latent_dim: vae_latent = args.latent_dim

    vae = MotionVAE(in_dim=in_dim, latent_dim=vae_latent).to(device)
    vae.load_state_dict(ckpt["state_dict"])
    vae.eval()  # freeze
    for p in vae.parameters(): p.requires_grad = False

    # Data
    ds_tr = SignDataset(Path(args.train), Path(args.data_root))
    ds_va = SignDataset(Path(args.val), Path(args.data_root))

    def collate(items):
        return collate_and_resample(items, t_target)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Cond model operating in latent space
    cond_dim = 512  # text embedding size
    unet = CondUNet1D(latent_dim=vae_latent, context_dim=cond_dim).to(device)
    ddpm = SimpleDDPM(unet, latent_dim=vae_latent, timesteps=1000, device=device)
    optim = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        unet.train()
        tr_loss = 0.0; n_tr = 0
        for batch in dl_tr:
            x = batch["pose"].to(device)      # [B, T, J, C]
            x, _ = flatten_TJC(x)             # [B, D]
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z0 = vae.reparam(mu, logvar)  # [B, L]

            texts = [ t if (t and isinstance(t, str) and len(t)>0) else (g if (g and isinstance(g, str)) else "") for t,g in zip(batch["text"], batch["gloss"]) ]
            ctx = encode_text(texts).to(device).float()

            loss = ddpm.training_loss(z0, ctx)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optim.step()

            bs = x.size(0); tr_loss += loss.item()*bs; n_tr += bs

        tr_loss /= n_tr

        # Simple val (same objective)
        unet.eval()
        va_loss = 0.0; n_va = 0
        with torch.no_grad():
            for batch in dl_va:
                x = batch["pose"].to(device)
                x, _ = flatten_TJC(x)
                mu, logvar = vae.encode(x); z0 = vae.reparam(mu, logvar)
                texts = [ t if (t and isinstance(t, str) and len(t)>0) else (g if (g and isinstance(g, str)) else "") for t,g in zip(batch["text"], batch["gloss"]) ]
                ctx = encode_text(texts).to(device).float()
                loss = ddpm.training_loss(z0, ctx)
                bs = x.size(0); va_loss += loss.item()*bs; n_va += bs
        va_loss /= n_va

        print(f"[E{epoch:03d}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

        # Save
        ck = out / f"ldm_e{epoch:03d}_val{va_loss:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "state_dict": unet.state_dict(),
            "latent_dim": vae_latent,
            "cond_dim": cond_dim,
            "t_target": t_target,
            "vae_ckpt": str(args.vae_ckpt),
        }, ck)
        if epoch == 1 or va_loss == min(va_loss, tr_loss):
            best = out / "ldm_latest.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": unet.state_dict(),
                "latent_dim": vae_latent,
                "cond_dim": cond_dim,
                "t_target": t_target,
                "vae_ckpt": str(args.vae_ckpt),
            }, best)

if __name__ == "__main__":
    main()
