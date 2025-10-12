
# Offline Real-Time (Same Machine)

This kit runs the **AI generator** (VAE+LDM) and the **avatar** on the same PC.

## 1) Python server (local)
```bash
# in your venv
pip install fastapi uvicorn sentence-transformers torch

python server_realtime.py \
  --vae_ckpt /path/to/checkpoints_vae/vae_best.pt \
  --ldm_ckpt /path/to/checkpoints_ldm/ldm_latest.pt \
  --host 127.0.0.1 --port 8000
```

- WebSocket: `ws://127.0.0.1:8000/ws/motion`
- Request JSON: `{"text":"안녕하세요","fps":30}`
- Streamed frames: `{"t":0, "pose":[[x,y,z], ...]}` (J rows, C=2/3 cols)

## 2) Unity (same PC)
- Add `Unity_MotionStreamClient.cs` to a GameObject.
- Set `wsUrl` to `ws://127.0.0.1:8000/ws/motion`.
- Prepare a few **target Transforms** (empties) for key joints and assign into `jointTargets[]`:
  - Hips, Spine, Neck/Head, L/R Shoulder/Elbow/Wrist (hands), L/R UpperLeg/Knee/Ankle (feet)
- Use **Animation Rigging** constraints (TwoBoneIK, etc.) to bind your avatar bones to these targets.

> Tip: The generated pose is **root-centered & height-normalized**. Position targets relative to your avatar root,
then adjust a one-time offset/scale if needed.

## Mapping
Index order must match your dataset joint order. Start with upper-body for sign (Hips..Wrist).

## Latency
- Single-machine loopback; end-to-end a few ms on GPU (more on CPU). If needed, lower the sampling steps,
or pre-generate clips you use often.
