# train_text2pose.py
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset_retrieval import SignTextPoseDataset
from dataloader_utils import collate_sign_text
from models import SignTextRetrievalModel, clip_contrastive_loss

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--text_model", default="klue/roberta-base")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Dataset / DataLoader
    train_ds = SignTextPoseDataset(args.train_csv, args.data_root)
    val_ds   = SignTextPoseDataset(args.val_csv,   args.data_root)

    # J, C 추론 (첫 샘플)
    sample = train_ds[0]
    J = sample["pose"].shape[1]
    C = sample["pose"].shape[2]
    print(f"[INFO] Pose shape example: T={sample['pose'].shape[0]}, J={J}, C={C}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_sign_text,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_sign_text,
        pin_memory=True,
    )

    # 2) Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    model = SignTextRetrievalModel(J=J, C=C,
                                   d_model=256, d_out=256,
                                   text_model_name=args.text_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            pose = batch["pose"].to(device)              # [B,T,J,C]
            pose_mask = batch["pose_mask"].to(device)    # [B,T]
            texts = batch["texts"]

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=64,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            p_emb, t_emb = model(
                pose=pose,
                pose_mask=pose_mask,
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )

            loss = clip_contrastive_loss(p_emb, t_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                avg = running_loss / args.log_interval
                print(f"[TRAIN] epoch={epoch} step={global_step} loss={avg:.4f}")
                running_loss = 0.0

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                pose = batch["pose"].to(device)
                pose_mask = batch["pose_mask"].to(device)
                texts = batch["texts"]

                enc = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=64,
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                p_emb, t_emb = model(
                    pose=pose,
                    pose_mask=pose_mask,
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                loss = clip_contrastive_loss(p_emb, t_emb)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= max(1, n_val_batches)
        print(f"[VAL] epoch={epoch} val_loss={val_loss:.4f}")

        # best model 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "model_state": model.state_dict(),
                "J": J,
                "C": C,
                "text_model": args.text_model,
            }, ckpt_path)
            print(f"[SAVE] best model -> {ckpt_path}")

if __name__ == "__main__":
    main()
