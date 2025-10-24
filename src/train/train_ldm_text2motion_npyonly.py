#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
NPY-only Text-to-Motion LDM training.

- Reads CSVs with 'keypoint_npy' (absolute or data_root-relative)
- Resamples each motion to --t_target frames, pads J to batch max
- Encodes flattened motion with a pretrained VAE encoder (mu as latent)
- Trains a DDPM-style noise predictor (epsilon prediction) in VAE latent space
- Optional text conditioning via SentenceTransformer (if available)

Examples (Windows, GPU #1):
  python src\train\train_ldm_text2motion_npyonly.py ^
    --train data\dataset\train_withnpy.csv ^
    --val   data\dataset\val_withnpy.csv ^
    --data_root C:\Users\Bry\ub\ubiplay ^
    --vae_ckpt runs\checkpoints_vae\vae_best.pt ^
    --epochs 50 --batch_size 32 --t_target 60 ^
    --gpu 1 --num_workers 2 --log_every 200 ^
    --out_dir runs\checkpoints_ldm
"""

import argparse, math, time
from functools import partial
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------- Utils: device ----------------------
def pick_device(gpu_arg: str) -> torch.device:
    ga = str(gpu_arg).strip().lower()
    if ga in ("-1", "cpu", "none"):
        return torch.device("cpu")
    if ga == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        idx = int(ga)
        dev = torch.device(f"cuda:{idx}")
    except ValueError:
        dev = torch.device(gpu_arg)  # e.g., "cuda:1"
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to CPU.")
        return torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    return dev

# ---------------------- Data utils -------------------------
def resample_T(x: np.ndarray, t_out: int) -> np.ndarray:
    """x: [T,J,C] -> [t_out,J,C] (per (J,C) linear)"""
    T, J, C = x.shape
    if T == t_out: return x
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
    if J == J_max: return x
    y = np.zeros((T, J_max, C), dtype=x.dtype)
    y[:, :J, :] = x
    return y

def flatten_TJC(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int,int]]:
    """x: [B,T,J,C] -> flat: [B, T*J*C]."""
    B, T, J, C = x.shape
    return x.reshape(B, T*J*C), (T, J, C)

# ---------------------- Dataset (NPY + optional text) ------
class NpyTextDataset(Dataset):
    def __init__(self, csv_path: Path, data_root: Path, text_cols: List[str]):
        self.root = Path(data_root)
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "keypoint_npy" not in df.columns:
            raise RuntimeError("CSV에 keypoint_npy가 필요합니다. index_to_npy.py 결과 CSV를 사용하세요.")
        ok = df["keypoint_npy"].notna() & (df["keypoint_npy"].astype(str).str.len() > 0)
        self.df = df[ok].reset_index(drop=True)
        self.text_cols = text_cols

    def __len__(self): return len(self.df)

    def _row_text(self, row) -> str:
        # 우선순위대로 이어 붙이기 (없으면 빈 문자열)
        vals = []
        for col in self.text_cols:
            if col in self.df.columns:
                v = row.get(col)
                if isinstance(v, str) and len(v):
                    vals.append(v)
        return " ".join(vals) if vals else ""

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = str(row["keypoint_npy"])
        p = Path(path) if Path(path).is_absolute() else (self.root / path)
        X = np.load(p)  # [T,J,C], float32
        text = self._row_text(row)
        return {"pose": torch.from_numpy(X).float(), "text": text}

def collate_fn(items, t_target: int):
    arrs = [it["pose"].numpy() for it in items]  # CPU side
    texts = [it["text"] for it in items]
    J_max = max(a.shape[1] for a in arrs)
    out = []
    for a in arrs:
        a2 = resample_T(a, t_target)
        a3 = pad_J(a2, J_max)
        out.append(a3)
    X = torch.from_numpy(np.stack(out, axis=0)).float()  # [B,T,Jmax,C]
    return {"pose": X, "text": texts}

# ---------------------- VAE (encoder only) -----------------
class MotionVAE(nn.Module):
    """Must match the VAE used previously (encoder mu/logvar heads)."""
    def __init__(self, in_dim: int, latent_dim: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU()
        )
        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def encode(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x_flat)
        return self.mu(h), self.logvar(h)

# ---------------------- Diffusion pieces -------------------
def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding like Stable Diffusion / DDPM.
    timesteps: [B], int64, in [0, T-1]
    return: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / max(half,1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1))
    return emb

class EpsMLP(nn.Module):
    """Noise predictor εθ(z_t, t, cond)."""
    def __init__(self, latent_dim: int, t_emb_dim: int = 128, cond_dim: int = 0, hidden: int = 1024):
        super().__init__()
        self.fc_in = nn.Linear(latent_dim + t_emb_dim + cond_dim, hidden)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
        self.t_proj = nn.Linear(t_emb_dim, t_emb_dim)  # small affine on t-embed
        self.cond_dim = cond_dim
        self.t_emb_dim = t_emb_dim
        self.latent_dim = latent_dim

    def forward(self, zt: torch.Tensor, t_emb: torch.Tensor, cond: Optional[torch.Tensor] = None):
        # zt: [B,L], t_emb: [B,t_emb_dim], cond: [B,cond_dim] or None
        te = self.t_proj(t_emb)
        if cond is None:
            x = torch.cat([zt, te], dim=-1)
        else:
            x = torch.cat([zt, te, cond], dim=-1)
        h = self.fc_in(x)
        return self.mlp(h)

def make_beta_schedule(n_steps: int, beta_start=1e-4, beta_end=2e-2):
    betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_cumprod

# ---------------------- Optional text encoder ---------------
def build_text_encoder(model_name: str, device: torch.device):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        print("[INFO] sentence-transformers not found. Falling back to unconditional model.")
        return None, 0
    try:
        enc = SentenceTransformer(model_name, device=str(device))
        dim = enc.get_sentence_embedding_dimension()
        print(f"[INFO] text encoder loaded: {model_name} (dim={dim})")
        return enc, int(dim)
    except Exception as e:
        print(f"[WARN] failed to load text encoder ({e}). Using unconditional.")
        return None, 0

def encode_text_batch(encoder, texts: List[str], device: torch.device, dim: int):
    if encoder is None or dim <= 0:
        return None
    # returns torch.FloatTensor [B,dim]
    try:
        emb = encoder.encode(texts, convert_to_tensor=True, device=str(device), show_progress_bar=False)
        return emb.float()
    except Exception as e:
        print(f"[WARN] text encoding failed ({e}); using unconditional for this batch.")
        return None

# ---------------------- Train loop --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--vae_ckpt", required=True, help="Path to VAE checkpoint (.pt)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--t_target", type=int, default=60)
    ap.add_argument("--latent_dim", type=int, default=256, help="(optional) override VAE latent_dim")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--diff_steps", type=int, default=1000)
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=2e-2)
    ap.add_argument("--out_dir", type=Path, default=Path("runs/checkpoints_ldm"))
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--gpu", type=str, default="auto")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--text_cols", type=str, default="text,gloss",
                    help="comma-separated columns to use as text condition (if present)")
    ap.add_argument("--text_model", type=str,
                    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    help="SentenceTransformer model name (optional)")
    args = ap.parse_args()

    device = pick_device(args.gpu)
    print(f"[INFO] device = {device}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Dataset / Loader
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    ds_tr = NpyTextDataset(Path(args.train), Path(args.data_root), text_cols)
    ds_va = NpyTextDataset(Path(args.val),   Path(args.data_root), text_cols)
    collate = partial(collate_fn, t_target=args.t_target)
    pin_mem = (device.type == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=pin_mem,
                       collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=pin_mem,
                       collate_fn=collate)

    print(f"[DATA] train={len(ds_tr)}  val={len(ds_va)}  batch={args.batch_size}  workers={args.num_workers}")

    # Peek first batch
    X0 = next(iter(dl_tr))
    pose0 = X0["pose"]
    print(f"[SHAPE] first train batch pose = {tuple(pose0.shape)}")  # [B,T,J,C]
    in_dim = pose0.shape[1] * pose0.shape[2] * pose0.shape[3]
    print(f"[INFO] in_dim (for VAE) = {in_dim}")

    # VAE encoder load
    vae = MotionVAE(in_dim=in_dim, latent_dim=args.latent_dim).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print(f"[VAE] loaded. missing={len(missing)} unexpected={len(unexpected)}")
    if "latent_dim" in ckpt:
        if args.latent_dim != ckpt["latent_dim"]:
            print(f"[WARN] overriding latent_dim from {ckpt['latent_dim']} -> {args.latent_dim}")
    latent_dim = args.latent_dim

    # Text encoder (optional)
    text_enc, cond_dim = build_text_encoder(args.text_model, device)

    # Epsilon predictor
    epsnet = EpsMLP(latent_dim=latent_dim, t_emb_dim=128, cond_dim=cond_dim, hidden=1024).to(device)
    opt = torch.optim.AdamW(epsnet.parameters(), lr=args.lr)

    # Diffusion schedule
    betas = torch.linspace(args.beta_start, args.beta_end, args.diff_steps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)                           # [T]
    sqrt_alphacum = torch.sqrt(alphas_cumprod)                              # [T]
    sqrt_one_minus_alphacum = torch.sqrt(1.0 - alphas_cumprod)              # [T]

    best = float("inf")
    for e in range(1, args.epochs + 1):
        # ---------------- Train ----------------
        epsnet.train(); vae.eval()
        t0 = time.time()
        seen = 0; loss_sum = 0.0
        total_batches = len(dl_tr)
        print(f"\n[Epoch {e:03d}/{args.epochs}] start ... (batches={total_batches})")

        for bi, batch in enumerate(dl_tr, 1):
            pose = batch["pose"].to(device, non_blocking=True)          # [B,T,J,C]
            B = pose.size(0)
            x_flat, _ = flatten_TJC(pose)                               # [B,in_dim]
            with torch.no_grad():
                mu, logvar = vae.encode(x_flat)                         # [B,L]
                z0 = mu                                                # deterministic latent (no reparam)

            # pick random timesteps per sample
            t = torch.randint(0, args.diff_steps, (B,), device=device, dtype=torch.long)  # [B]
            eps = torch.randn_like(z0)

            at = sqrt_alphacum[t].unsqueeze(1)                          # [B,1]
            att = sqrt_one_minus_alphacum[t].unsqueeze(1)               # [B,1]
            zt = at * z0 + att * eps                                    # forward diffusion

            # t embedding
            t_emb = timestep_embedding(t, dim=128)                      # [B,128]

            # cond
            cond = encode_text_batch(text_enc, batch["text"], device, cond_dim)
            # predict eps
            eps_hat = epsnet(zt, t_emb, cond)                           # [B,L]

            loss = nn.MSELoss()(eps_hat, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(epsnet.parameters(), 1.0)
            opt.step()

            bs = B
            loss_sum += float(loss.item()) * bs
            seen += bs

            if (bi % args.log_every == 0) or bi == 1 or bi == total_batches:
                eta = (time.time() - t0) / bi * (total_batches - bi)
                print(f"[E{e:03d}] {bi:>5}/{total_batches}  loss={float(loss.item()):.4f}  ETA~{eta:,.1f}s")

        tr_loss = loss_sum / max(1, seen)

        # ---------------- Val ----------------
        epsnet.eval()
        v_seen = 0; v_sum = 0.0
        with torch.no_grad():
            for vb, batch in enumerate(dl_va, 1):
                pose = batch["pose"].to(device, non_blocking=True)
                B = pose.size(0)
                x_flat, _ = flatten_TJC(pose)
                mu, logvar = vae.encode(x_flat)
                z0 = mu
                t = torch.randint(0, args.diff_steps, (B,), device=device, dtype=torch.long)
                eps = torch.randn_like(z0)
                at = sqrt_alphacum[t].unsqueeze(1)
                att = sqrt_one_minus_alphacum[t].unsqueeze(1)
                zt = at * z0 + att * eps
                t_emb = timestep_embedding(t, dim=128)
                cond = encode_text_batch(text_enc, batch["text"], device, cond_dim)
                eps_hat = epsnet(zt, t_emb, cond)
                loss = nn.MSELoss()(eps_hat, eps)
                v_sum += float(loss.item()) * B
                v_seen += B
        v_loss = v_sum / max(1, v_seen)

        dt = time.time() - t0
        print(f"[Epoch {e:03d}] done in {dt:.1f}s | train_loss={tr_loss:.4f}  val_loss={v_loss:.4f}")

        # ---------------- Save ----------------
        ck = out / f"ldm_e{e:03d}_val{v_loss:.4f}.pt"
        torch.save({
            "epoch": e,
            "epsnet": epsnet.state_dict(),
            "vae_in_dim": in_dim,
            "latent_dim": latent_dim,
            "t_target": int(pose0.shape[1]),
            "diff_steps": args.diff_steps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "text_dim": cond_dim
        }, ck)
        print(f"[SAVE] {ck}")
        if v_loss < best:
            best = v_loss
            bestp = out / "ldm_best.pt"
            torch.save({
                "epoch": e,
                "epsnet": epsnet.state_dict(),
                "vae_in_dim": in_dim,
                "latent_dim": latent_dim,
                "t_target": int(pose0.shape[1]),
                "diff_steps": args.diff_steps,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
                "text_dim": cond_dim
            }, bestp)
            print(f"[BEST] updated -> {bestp}")

    print("[DONE] LDM training finished.")

if __name__ == "__main__":
    main()
