#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_diffusion.py

데이터셋 구조에 맞춘 3D 수어 생성 Diffusion 모델 학습 스크립트.
- train_withnpy.csv / val_withnpy.csv 사용
- .npy (포즈)와 _morpheme.json (형태소)을 로드
- 형태소(gloss)를 조건(condition)으로 사용
- 가변 길이 시퀀스를 패딩(padding)하여 배치 처리
"""

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# --- 경고 메시지 무시 (선택 사항) ---
warnings.filterwarnings("ignore", category=UserWarning)

# === 1. 유틸리티 함수 (dataset_sen.py에서 가져옴) ===

def _read_json(p: Path) -> Dict[str, Any]:
    """ .json 파일을 읽어 딕셔너리로 반환 """
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # 파일이 없거나 깨진 경우
        return {}

def _load_text_from_morpheme(morph_path: Path) -> Dict[str, Optional[str]]:
    """ morpheme.json에서 text와 gloss(형태소)를 추출 """
    d = _read_json(morph_path)
    # TTA 표준 및 일반적인 키 값들을 순서대로 탐색
    text = d.get("text") or d.get("korean") or d.get("sentence") or d.get("문장")
    gloss = d.get("gloss") or d.get("글로스") or d.get("morpheme") or d.get("형태소")
    return {"text": text, "gloss": gloss}

# === 2. 커스텀 데이터셋 ===

class SignDiffusionDataset(Dataset):
    """
    _withnpy.csv 파일을 읽어 .npy 포즈 데이터와 .json 형태소(gloss)를 로드하는 데이터셋
    """
    def __init__(self, index_csv: Path, data_root: Path, vocab: Dict[str, int]):
        self.data_root = Path(data_root)
        try:
            self.df = pd.read_csv(index_csv, encoding="utf-8-sig")
        except Exception as e:
            print(f"Error reading CSV {index_csv}: {e}")
            raise
            
        self.vocab = vocab
        self.unk_token_id = vocab.get("<UNK>", 0)

        # .npy 파일과 .json 파일이 모두 존재하는지 사전 필터링
        self.df = self._filter_valid_rows(self.df)
        print(f"Loaded {index_csv.name}. Valid items: {len(self.df)}")

    def _filter_valid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """ .npy와 morpheme.json 경로가 유효한 행만 필터링 """
        
        # NPY 파일 경로 확인 (sanity_check.py 로직 참조)
        if "keypoint_npy" not in df.columns:
            raise RuntimeError("CSV에 keypoint_npy 컬럼이 없습니다.")
        ok_npy = df["keypoint_npy"].notna() & (df["keypoint_npy"].astype(str).str.len() > 0)
        
        # Morpheme JSON 파일 경로 확인
        if "morpheme_json" not in df.columns:
            raise RuntimeError("CSV에 morpheme_json 컬럼이 없습니다.")
        ok_morph = df["morpheme_json"].notna() & (df["morpheme_json"].astype(str).str.len() > 0)
        
        # 형태소(gloss) 값 확인
        ok_gloss = df["gloss"].notna() & (df["gloss"].astype(str).str.len() > 0)

        return df[ok_npy & ok_morph & ok_gloss].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, p_str: str) -> Path:
        """ CSV의 경로를 data_root 기준으로 변환 """
        p = Path(p_str)
        if p.is_absolute():
            return p
        return self.data_root / p

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        row = self.df.iloc[idx]
        
        # 1. 포즈(.npy) 로드
        try:
            npy_path = self._resolve_path(row["keypoint_npy"])
            pose_data = np.load(npy_path) # [T, J, C]
        except Exception as e:
            print(f"Warning: Failed to load npy {npy_path}: {e}")
            return self(idx + 1 if idx + 1 < len(self) else 0) # 문제 시 다음 샘플

        # 2. 형태소(gloss) 로드
        # CSV에 'gloss' 컬럼이 미리 빌드되어 있다고 가정 (build_index/split_index가 생성)
        # 만약 없다면, morpheme_json을 직접 읽어야 함
        gloss = row.get("gloss")
        if not isinstance(gloss, str):
             try:
                 morph_path = self._resolve_path(row["morpheme_json"])
                 gloss = _load_text_from_morpheme(morph_path).get("gloss")
                 if not isinstance(gloss, str):
                     raise ValueError("Gloss is not a string")
             except Exception as e:
                 print(f"Warning: Failed to load gloss {row['morpheme_json']}: {e}")
                 return self(idx + 1 if idx + 1 < len(self) else 0)

        # 3. Gloss를 ID로 변환
        gloss_id = self.vocab.get(gloss, self.unk_token_id)
            
        return {
            "pose": torch.from_numpy(pose_data).float(), # [T, J, C]
            "gloss_id": torch.tensor(gloss_id, dtype=torch.long),
            "stem": row.get("stem", "")
        }

def build_vocabulary(train_csv_path: Path) -> Dict[str, int]:
    """ 훈련 CSV의 'gloss' 컬럼을 읽어 어휘집(vocabulary) 구축 """
    print(f"Building vocabulary from {train_csv_path.name}...")
    df = pd.read_csv(train_csv_path, encoding="utf-8-sig")
    
    if "gloss" not in df.columns:
        print("Warning: 'gloss' column not found in CSV. Using morpheme_json (slower).")
        # 이 경우, Dataset에서 매번 json을 읽어야 하므로 비효율적
        # 여기서는 build_index.py가 gloss를 CSV에 추가했다고 가정
        return {"<PAD>": 0, "<UNK>": 1} # 임시

    glosses = df["gloss"].dropna().unique()
    vocab = {"<PAD>": 0, "<UNK>": 1} # 0=패딩, 1=알수없음
    for g in glosses:
        if g not in vocab:
            vocab[g] = len(vocab)
            
    print(f"Vocabulary built. Size: {len(vocab)} tokens.")
    return vocab


def collate_fn_pad(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    가변 길이의 시퀀스를 패딩하여 배치 생성
    """
    # 1. 샘플이 None인 경우 필터링
    batch = [s for s in batch if s is not None]
    if not batch:
        return {} # 빈 배치

    # 2. 포즈 데이터 패딩
    poses = [s["pose"] for s in batch]
    lengths = torch.tensor([len(p) for p in poses], dtype=torch.long)
    
    # [T, J, C] -> [B, T, J, C]
    # pad_sequence는 (T, B, *)를 기대하므로, batch_first=True 사용
    padded_poses = pad_sequence(poses, batch_first=True, padding_value=0.0)
    
    # 3. 패딩 마스크 생성 (1 = 실제 데이터, 0 = 패딩)
    # (B, T)
    max_len = padded_poses.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    
    # 4. 나머지 데이터 스택
    gloss_ids = torch.stack([s["gloss_id"] for s in batch])
    
    return {
        "pose": padded_poses,     # (B, T, J, C)
        "mask": mask,             # (B, T)
        "gloss_id": gloss_ids,  # (B)
        "lengths": lengths        # (B)
    }


# === 3. Diffusion 모델 및 로직 ===

class PositionalEmbedding(nn.Module):
    """ Sinusoidal Positional Embedding (Timestep 임베딩용) """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embed = math.log(10000) / (half_dim - 1)
        embed = torch.exp(torch.arange(half_dim, device=device) * -embed)
        embed = t[:, None] * embed[None, :]
        embed = torch.cat((embed.sin(), embed.cos()), dim=-1)
        return embed

class SimpleDenoisingModel(nn.Module):
    """
    간단한 MLP 기반 Diffusion Denoising 모델
    - 입력: Noised Pose (B, T, J*C), Timestep (B), Gloss ID (B)
    - 출력: Predicted Noise (B, T, J*C)
    """
    def __init__(self, pose_dim, vocab_size, embed_dim=256, max_len=512):
        super().__init__()
        self.pose_dim = pose_dim # J * C
        
        # 임베딩
        self.time_embed = nn.Sequential(
            PositionalEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.gloss_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pose_pos_embed = nn.Embedding(max_len, embed_dim) # 각 프레임 위치

        # Denoising MLP
        self.in_proj = nn.Linear(pose_dim, embed_dim)
        self.blocks = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.out_proj = nn.Linear(embed_dim, pose_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, gloss_id: torch.Tensor) -> torch.Tensor:
        # x_t: (B, T, J*C)
        # t: (B)
        # gloss_id: (B)
        B, T, _ = x_t.shape
        
        # 1. 임베딩
        t_emb = self.time_embed(t)          # (B, E)
        g_emb = self.gloss_embed(gloss_id)  # (B, E)
        
        # (B, E) -> (B, T, E)로 확장
        cond_emb = (t_emb + g_emb).unsqueeze(1).expand(-1, T, -1)
        
        # 2. 위치 임베딩
        pos_ids = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1) # (B, T)
        pos_emb = self.pose_pos_embed(pos_ids) # (B, T, E)

        # 3. Denoising
        h = self.in_proj(x_t) # (B, T, E)
        
        # 조건과 위치 정보 결합
        h = h + cond_emb + pos_emb
        
        h = self.blocks(h)    # (B, T, E)
        out = self.out_proj(h)  # (B, T, J*C)
        
        return out


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# === 4. 메인 학습 로직 ===

def main(args):
    # --- 1. 설정 ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv = Path(args.train)
    val_csv = Path(args.val)
    data_root = Path(args.data_root)

    # --- 2. Diffusion 설정 ---
    timesteps = args.timesteps
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    # x_t 계산에 필요한 값들
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    def get_noisy_pose(x_start, t, noise):
        """ Forward process (q): x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise """
        # x_start: (B, T, D)
        # t: (B)
        # noise: (B, T, D)
        B, T, D = x_start.shape
        
        # t에 해당하는 alpha_bar 값을 가져와서 (B, 1, 1) 형태로 변환
        sqrt_alpha_t = sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        
        return (sqrt_alpha_t * x_start) + (sqrt_one_minus_alpha_t * noise)

    # --- 3. 데이터 로더 ---
    vocab = build_vocabulary(train_csv)
    vocab_size = len(vocab)
    
    train_dataset = SignDiffusionDataset(train_csv, data_root, vocab)
    val_dataset = SignDiffusionDataset(val_csv, data_root, vocab)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_pad,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_pad,
        pin_memory=True
    )

    # --- 4. 모델, 옵티마이저, 손실 함수 ---
    
    # J, C 값 확인 (첫 번째 데이터 샘플 기준)
    try:
        sample = train_dataset[0]["pose"]
        _, J, C = sample.shape
        pose_dim = J * C
        max_len_sample = max(s["pose"].shape[0] for s in train_dataset.df.index.map(lambda i: train_dataset[i]))
        max_len = min(max_len_sample, args.max_seq_len) # 너무 길면 제한
        print(f"Detected pose shape: T, J={J}, C={C}. pose_dim={pose_dim}. Max sequence length: {max_len}")
    except Exception as e:
        print(f"Could not determine pose shape from dataset: {e}")
        return

    model = SimpleDenoisingModel(
        pose_dim=pose_dim, 
        vocab_size=vocab_size, 
        embed_dim=args.embed_dim,
        max_len=max_len
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss(reduction='none') # 패딩 마스크 적용을 위해 reduction='none'

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 5. 학습 루프 ---
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # --- Training ---
        model.train()
        train_loss_total = 0.0
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False)
        
        for batch in train_pbar:
            if not batch: continue # 빈 배치 스킵
                
            pose = batch["pose"].to(device)         # (B, T, J, C)
            mask = batch["mask"].to(device)         # (B, T)
            gloss_id = batch["gloss_id"].to(device) # (B)
            
            B, T, J, C = pose.shape
            
            # (B, T, J, C) -> (B, T, J*C)
            pose_flat = pose.view(B, T, -1)
            mask_flat = mask.unsqueeze(-1) # (B, T, 1)

            # 1. 타임스텝, 노이즈 생성
            t = torch.randint(0, timesteps, (B,), device=device).long()
            noise = torch.randn_like(pose_flat) # (B, T, J*C)
            
            # 2. Noisy Pose 생성 (Forward process)
            x_t = get_noisy_pose(pose_flat, t, noise)
            
            # 3. 노이즈 예측 (Model)
            predicted_noise = model(x_t, t, gloss_id)
            
            # 4. 손실 계산 (Loss)
            loss = loss_fn(predicted_noise, noise) # (B, T, J*C)
            
            # 5. 마스크 적용 (패딩된 부분은 손실에서 제외)
            loss = loss * mask_flat # (B, T, J*C)
            
            # 배치 평균
            # .sum() / mask.sum() -> 픽셀(관절) 단위 평균
            loss = loss.sum() / (mask_flat.sum() * pose_dim + 1e-8) 

            # 6. 역전파
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient Clipping
            optimizer.step()
            
            train_loss_total += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_total / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.6f}")

        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        val_pbar = tqdm(val_loader, desc=f"Validate Epoch {epoch+1}", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                if not batch: continue
                    
                pose = batch["pose"].to(device)
                mask = batch["mask"].to(device)
                gloss_id = batch["gloss_id"].to(device)
                
                B, T, J, C = pose.shape
                pose_flat = pose.view(B, T, -1)
                mask_flat = mask.unsqueeze(-1)

                t = torch.randint(0, timesteps, (B,), device=device).long()
                noise = torch.randn_like(pose_flat)
                
                x_t = get_noisy_pose(pose_flat, t, noise)
                predicted_noise = model(x_t, t, gloss_id)
                
                loss = loss_fn(predicted_noise, noise)
                loss = loss * mask_flat
                loss = loss.sum() / (mask_flat.sum() * pose_dim + 1e-8)
                
                val_loss_total += loss.item()
                val_pbar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss:.6f}")
        
        # --- Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = out_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'vocab': vocab, # 어휘집 저장
                'pose_dim': pose_dim,
                'embed_dim': args.embed_dim,
                'max_len': max_len
            }, ckpt_path)
            print(f"New best model saved to {ckpt_path} (Loss: {best_val_loss:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training for 3D Sign Pose Generation")
    
    # --- 요청하신 옵션 ---
    parser.add_argument('--train', type=str, required=True, help='Path to train_withnpy.csv')
    parser.add_argument('--val', type=str, required=True, help='Path to val_withnpy.csv')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset (e.g., C:Users...ubiplay)')
    parser.add_argument('--epochs', type=int, default=200, help='Total number of epochs (추천: 200+, 테스트: 30)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use (default: 0)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--out_dir', type=str, default='runs/checkpoints_diffusion', help='Directory to save checkpoints')

    # --- 모델 및 학습 하이퍼파라미터 ---
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension for model')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length to cap poses')

    args = parser.parse_args()
    
    # 사용자가 제공한 예시 옵션 출력 (확인용)
    print("--- Running with options ---")
    print(f"  --train {args.train}")
    print(f"  --val {args.val}")
    print(f"  --data_root {args.data_root}")
    print(f"  --epochs {args.epochs}")
    print(f"  --batch_size {args.batch_size}")
    print(f"  --gpu {args.gpu}")
    print(f"  --num_workers {args.num_workers}")
    print(f"  --out_dir {args.out_dir}")
    print("----------------------------")

    main(args)
