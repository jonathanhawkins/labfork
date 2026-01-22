#!/usr/bin/env python3
"""
Train prosody encoder on RAVDESS emotional speech dataset.
Designed for RTX 4090 with larger batches.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime


class ProsodyDataset(Dataset):
    """Dataset for prosody training from manifest."""

    def __init__(self, manifest_path: str, max_audio_len: int = 24000 * 10):
        with open(manifest_path) as f:
            self.samples = json.load(f)

        self.base_dir = Path(manifest_path).parent
        self.max_audio_len = max_audio_len

        # Filter samples with valid prosody
        self.samples = [s for s in self.samples if s.get("prosody")]
        print(f"Loaded {len(self.samples)} samples with prosody annotations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        audio_path = self.base_dir / sample["audio_path"]
        waveform, sr = torchaudio.load(str(audio_path))

        # Resample if needed
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate
        if waveform.shape[1] > self.max_audio_len:
            waveform = waveform[:, :self.max_audio_len]
        elif waveform.shape[1] < self.max_audio_len:
            padding = self.max_audio_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Extract prosody targets
        prosody = sample["prosody"]

        # Semantic features (emotion one-hot + intensity)
        emotion_map = {
            "neutral": 0, "calm": 1, "happy": 2, "sad": 3,
            "angry": 4, "fearful": 5, "surprised": 6, "excited": 7
        }
        emotion = prosody.get("semantic", {}).get("emotion", "neutral")
        emotion_idx = emotion_map.get(emotion, 0)
        emotion_onehot = torch.zeros(8)
        emotion_onehot[emotion_idx] = 1.0
        intensity = prosody.get("semantic", {}).get("intensity", 0.5)
        semantic = torch.cat([emotion_onehot, torch.tensor([intensity])])

        # Acoustic features
        acoustic = prosody.get("acoustic", {})
        acoustic_vec = torch.tensor([
            acoustic.get("pitch_mean", 0.5),
            acoustic.get("pitch_std", 0.1),
            acoustic.get("energy", 0.5),
            acoustic.get("speaking_rate", 0.5),
        ])

        # Rhythm features
        rhythm = prosody.get("rhythm", {})
        rhythm_vec = torch.tensor([
            rhythm.get("pause_ratio", 0.2),
            rhythm.get("syllable_rate", 4.0) / 10.0,  # Normalize
        ])

        # Contour (64 points)
        contour = prosody.get("contour", {}).get("pitch_contour", [0.5] * 64)
        contour_vec = torch.tensor(contour[:64])
        if len(contour_vec) < 64:
            contour_vec = torch.nn.functional.pad(contour_vec, (0, 64 - len(contour_vec)), value=0.5)

        return {
            "audio": waveform.squeeze(0),
            "semantic": semantic.float(),
            "acoustic": acoustic_vec.float(),
            "rhythm": rhythm_vec.float(),
            "contour": contour_vec.float(),
            "emotion_idx": emotion_idx,
        }


class ProsodyEncoder(nn.Module):
    """
    Encoder that learns to predict prosody features from audio.
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Audio encoder (1D conv over waveform)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=160),  # ~25ms windows
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, hidden_dim, kernel_size=3, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Prosody prediction heads
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 9),  # 8 emotions + intensity
        )

        self.acoustic_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # pitch_mean, pitch_std, energy, speaking_rate
            nn.Sigmoid(),
        )

        self.rhythm_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # pause_ratio, syllable_rate
            nn.Sigmoid(),
        )

        self.contour_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),  # 64-point contour
            nn.Sigmoid(),
        )

        # Emotion classifier (for auxiliary loss)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
        )

    def forward(self, audio):
        # audio: (B, T)
        x = audio.unsqueeze(1)  # (B, 1, T)

        # Encode audio
        features = self.audio_encoder(x)  # (B, hidden_dim, 1)
        features = features.squeeze(-1)   # (B, hidden_dim)

        # Predict prosody
        semantic = self.semantic_head(features)
        acoustic = self.acoustic_head(features)
        rhythm = self.rhythm_head(features)
        contour = self.contour_head(features)
        emotion_logits = self.emotion_classifier(features)

        return {
            "semantic": semantic,
            "acoustic": acoustic,
            "rhythm": rhythm,
            "contour": contour,
            "emotion_logits": emotion_logits,
            "features": features,
        }


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    emotion_correct = 0
    emotion_total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        audio = batch["audio"].to(device)
        semantic_target = batch["semantic"].to(device)
        acoustic_target = batch["acoustic"].to(device)
        rhythm_target = batch["rhythm"].to(device)
        contour_target = batch["contour"].to(device)
        emotion_idx = batch["emotion_idx"].to(device)

        optimizer.zero_grad()

        outputs = model(audio)

        # Losses
        semantic_loss = nn.functional.mse_loss(outputs["semantic"], semantic_target)
        acoustic_loss = nn.functional.mse_loss(outputs["acoustic"], acoustic_target)
        rhythm_loss = nn.functional.mse_loss(outputs["rhythm"], rhythm_target)
        contour_loss = nn.functional.mse_loss(outputs["contour"], contour_target)
        emotion_loss = nn.functional.cross_entropy(outputs["emotion_logits"], emotion_idx)

        # Combined loss
        loss = semantic_loss + acoustic_loss + rhythm_loss + contour_loss + 0.5 * emotion_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # Track emotion accuracy
        pred_emotion = outputs["emotion_logits"].argmax(dim=1)
        emotion_correct += (pred_emotion == emotion_idx).sum().item()
        emotion_total += emotion_idx.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "emo_acc": f"{emotion_correct/emotion_total:.2%}"
        })

    return total_loss / len(dataloader), emotion_correct / emotion_total


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    emotion_correct = 0
    emotion_total = 0

    with torch.no_grad():
        for batch in dataloader:
            audio = batch["audio"].to(device)
            semantic_target = batch["semantic"].to(device)
            acoustic_target = batch["acoustic"].to(device)
            rhythm_target = batch["rhythm"].to(device)
            contour_target = batch["contour"].to(device)
            emotion_idx = batch["emotion_idx"].to(device)

            outputs = model(audio)

            semantic_loss = nn.functional.mse_loss(outputs["semantic"], semantic_target)
            acoustic_loss = nn.functional.mse_loss(outputs["acoustic"], acoustic_target)
            rhythm_loss = nn.functional.mse_loss(outputs["rhythm"], rhythm_target)
            contour_loss = nn.functional.mse_loss(outputs["contour"], contour_target)
            emotion_loss = nn.functional.cross_entropy(outputs["emotion_logits"], emotion_idx)

            loss = semantic_loss + acoustic_loss + rhythm_loss + contour_loss + 0.5 * emotion_loss
            total_loss += loss.item()

            pred_emotion = outputs["emotion_logits"].argmax(dim=1)
            emotion_correct += (pred_emotion == emotion_idx).sum().item()
            emotion_total += emotion_idx.size(0)

    return total_loss / len(dataloader), emotion_correct / emotion_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/prosody_training/manifest.json")
    parser.add_argument("--output", type=str, default="../models/prosody_encoder_ravdess")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load dataset
    dataset = ProsodyDataset(args.data)

    # Split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = ProsodyEncoder(hidden_dim=args.hidden_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_emotion_acc = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output: {output_dir}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.2%}, val_acc={val_acc:.2%}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_emotion_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, output_dir / "best.pt")
            print(f"  -> Saved best model (loss={val_loss:.4f}, acc={val_acc:.2%})")

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, output_dir / f"checkpoint_epoch{epoch}.pt")

    print(f"\nTraining complete!")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Best emotion accuracy: {best_emotion_acc:.2%}")


if __name__ == "__main__":
    main()
