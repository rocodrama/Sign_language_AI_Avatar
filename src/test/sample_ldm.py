
import argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

from utils import flatten_TJC
from ldm_stub import MotionVAE, CondUNet1D
# Optional text encoder
try:
    from sentence_transformers import SentenceTransformer
    HAVE_TXT = True
except Exception:
    HAVE_TXT = False

def timesteps_schedule(T=1000):
    return torch.linspace(1e-4, 0.02, T)

def encode_text(texts):
    if HAVE_TXT:
        enc = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
        z = enc.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return z
    else:
        return torch.zeros((len(texts), 512))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--ldm_ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out_npy", required=True, help="Output .npy for [T,J,C] motion")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    v = torch.load(args.vae_ckpt, map_location="cpu")
    vae = MotionVAE(in_dim=v["in_dim"], latent_dim=v["latent_dim"]).to(device)
    vae.load_state_dict(v["state_dict"]); vae.eval()
    T_target = v["t_target"]; J, C = v["joints"]

    # Load LDM
    l = torch.load(args.ldm_ckpt, map_location="cpu")
    unet = CondUNet1D(latent_dim=l["latent_dim"], context_dim=l["cond_dim"]).to(device)
    unet.load_state_dict(l["state_dict"]); unet.eval()

    # DDPM params
    timesteps = 1000
    betas = timesteps_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    # Sample in latent space
    ctx = encode_text([args.text]).to(device).float()
    z = torch.randn(1, l["latent_dim"], device=device)
    for t in reversed(range(timesteps)):
        ab_t = alpha_bar[t]
        eps = unet(z, t_embed=None, context=ctx)
        z = (z - (1 - alphas[t]) / torch.sqrt(1 - ab_t) * eps) / torch.sqrt(alphas[t])
        if t > 0:
            z = z + torch.sqrt(betas[t]) * torch.randn_like(z)

    # Decode to motion
    with torch.no_grad():
        motion_flat = vae.decode(z)  # [1, T*J*C]
    motion = motion_flat.view(T_target, J, C).cpu().numpy()
    np.save(args.out_npy, motion)
    print(f"Saved motion to {args.out_npy} shape={motion.shape}")

if __name__ == "__main__":
    main()
