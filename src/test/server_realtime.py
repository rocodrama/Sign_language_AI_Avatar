
import argparse, asyncio, json, time
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn

from ldm_stub import MotionVAE, CondUNet1D

try:
    from sentence_transformers import SentenceTransformer
    HAVE_TXT = True
except Exception:
    HAVE_TXT = False
    SentenceTransformer = None

def timesteps_schedule(T=1000):
    return torch.linspace(1e-4, 0.02, T)

def encode_text(texts):
    if HAVE_TXT:
        if not hasattr(encode_text, "_model"):
            encode_text._model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
        z = encode_text._model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return z
    else:
        return torch.zeros((len(texts), 512))

def load_models(vae_ckpt: str, ldm_ckpt: str, device: torch.device):
    v = torch.load(vae_ckpt, map_location="cpu")
    vae = MotionVAE(in_dim=v["in_dim"], latent_dim=v["latent_dim"]).to(device)
    vae.load_state_dict(v["state_dict"]); vae.eval()
    T_target = v["t_target"]; J, C = v["joints"]

    l = torch.load(ldm_ckpt, map_location="cpu")
    unet = CondUNet1D(latent_dim=l["latent_dim"], context_dim=l["cond_dim"]).to(device)
    unet.load_state_dict(l["state_dict"]); unet.eval()

    # diffusion params cached
    timesteps = 1000
    betas = timesteps_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    return vae, unet, (timesteps, betas, alphas, alpha_bar), T_target, J, C

def sample_motion(vae, unet, diff_params, text: str, device, latent_dim: int, T_target: int, J: int, C: int):
    timesteps, betas, alphas, alpha_bar = diff_params
    with torch.no_grad():
        ctx = encode_text([text]).to(device).float()
        z = torch.randn(1, latent_dim, device=device)
        for t in reversed(range(timesteps)):
            ab_t = alpha_bar[t]
            eps = unet(z, t_embed=None, context=ctx)
            z = (z - (1 - alphas[t]) / torch.sqrt(1 - ab_t) * eps) / torch.sqrt(alphas[t])
            if t > 0:
                z = z + torch.sqrt(betas[t]) * torch.randn_like(z)
        motion_flat = vae.decode(z)  # [1, T*J*C]
        motion = motion_flat.view(T_target, J, C).detach().cpu().numpy()
        return motion

def build_app(vae, unet, diff_params, latent_dim, T, J, C, device):
    app = FastAPI()

    @app.get("/")
    def info():
        return {"status": "ok", "T": T, "J": J, "C": C}

    @app.websocket("/ws/motion")
    async def ws_motion(ws: WebSocket):
        await ws.accept()
        try:
            payload = await ws.receive_text()
            req = json.loads(payload) if payload.strip().startswith("{") else {"text": payload}
            text = req.get("text", "")
            fps = int(req.get("fps", 30))
            # Generate once
            motion = sample_motion(vae, unet, diff_params, text, device, latent_dim, T, J, C)
            # Stream frame by frame
            dt = 1.0 / max(1, fps)
            for t in range(motion.shape[0]):
                frame = motion[t].tolist()  # [J][C]
                await ws.send_text(json.dumps({"t": t, "pose": frame}))
                await asyncio.sleep(dt)
            await ws.close()
        except Exception as e:
            try:
                await ws.close()
            except:
                pass

    return app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--ldm_ckpt", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    vae, unet, diff_params, T, J, C = load_models(args.vae_ckpt, args.ldm_ckpt, device)
    latent_dim = unet.net[0].in_features - 512  # (latent_dim + context_dim)

    app = build_app(vae, unet, diff_params, latent_dim, T, J, C, device)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
