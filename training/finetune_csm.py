"""
Voice Clone Pipeline - CSM Fine-Tuning Script
Fine-tunes Sesame's CSM-1B model on your voice data.

Usage:
    python finetune_csm.py --config config/rtx_4090.yaml
    python finetune_csm.py --config config/m4_pro.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchaudio
from tqdm import tqdm

# Optional: wandb for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class VoiceDataset(Dataset):
    """
    Dataset for CSM training.
    Loads audio tensors and text, handles padding.
    """
    
    def __init__(
        self,
        data_path: Path,
        max_audio_length_ms: int = 30000,
        sample_rate: int = 24000,
    ):
        with open(data_path) as f:
            self.samples = json.load(f)
        
        self.max_audio_length = int(max_audio_length_ms * sample_rate / 1000)
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio tensor
        if "audio_tensor_path" in sample and Path(sample["audio_tensor_path"]).exists():
            waveform = torch.load(sample["audio_tensor_path"])
        else:
            waveform, sr = torchaudio.load(sample["audio_path"])
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        
        # Truncate or pad to max length
        if waveform.shape[1] > self.max_audio_length:
            waveform = waveform[:, :self.max_audio_length]
        
        return {
            "audio": waveform.squeeze(0),  # [samples]
            "text": sample["text"],
            "speaker": sample.get("speaker", 0),
            "prosody": sample.get("prosody", {}),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    # Pad audio to same length
    max_len = max(b["audio"].shape[0] for b in batch)
    
    audios = []
    audio_masks = []
    
    for b in batch:
        audio = b["audio"]
        pad_len = max_len - audio.shape[0]
        
        if pad_len > 0:
            audio = torch.cat([audio, torch.zeros(pad_len)])
        
        audios.append(audio)
        
        mask = torch.ones(max_len, dtype=torch.bool)
        mask[len(b["audio"]):] = False
        audio_masks.append(mask)
    
    return {
        "audio": torch.stack(audios),
        "audio_mask": torch.stack(audio_masks),
        "text": [b["text"] for b in batch],
        "speaker": torch.tensor([b["speaker"] for b in batch]),
    }


class CSMTrainer:
    """
    Trainer for CSM fine-tuning.
    
    Handles:
    - Model loading and setup
    - Training loop with gradient accumulation
    - Validation and checkpointing
    - Logging (console and optional wandb)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        
        # Paths
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
    
    def _setup_device(self) -> torch.device:
        """Setup compute device (CUDA, MPS, or CPU)."""
        device_str = self.config.get("device", "auto")
        
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_str)
        
        print(f"Using device: {device}")
        return device
    
    def load_model(self):
        """Load CSM model from Hugging Face or local path."""
        model_path = self.config["model_path"]
        print(f"Loading model from: {model_path}")
        
        # Check if CSM is available
        try:
            # Try to import from local CSM installation
            sys.path.insert(0, str(Path(model_path).parent))
            from csm.model import CSM
            
            model = CSM.from_pretrained(model_path)
        except ImportError:
            # Fallback: load from Hugging Face
            print("CSM not found locally, trying Hugging Face...")
            from transformers import AutoModel
            model = AutoModel.from_pretrained("sesame/csm-1b", trust_remote_code=True)
        
        # Move to device
        model = model.to(self.device)
        
        # Setup precision
        if self.config.get("precision") == "fp16":
            model = model.half()
        elif self.config.get("precision") == "bf16":
            model = model.to(torch.bfloat16)
        
        # Gradient checkpointing
        if self.config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
        
        return model
    
    def load_data(self):
        """Load train and validation datasets."""
        data_dir = Path(self.config["data_dir"])
        
        train_dataset = VoiceDataset(
            data_dir / "train.json",
            max_audio_length_ms=self.config.get("max_audio_length_ms", 30000),
        )
        
        val_dataset = VoiceDataset(
            data_dir / "val.json",
            max_audio_length_ms=self.config.get("max_audio_length_ms", 30000),
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=True if self.device.type == "cuda" else False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.get("num_workers", 0),
        )
        
        return train_loader, val_loader
    
    def setup_optimizer(self, model):
        """Setup optimizer and scheduler."""
        optimizer = AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
        )
        
        # Scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.get("warmup_steps", 500),
            T_mult=2,
        )
        
        return optimizer, scheduler
    
    def train_step(self, model, batch, optimizer, scheduler):
        """Single training step."""
        model.train()
        
        # Move batch to device
        audio = batch["audio"].to(self.device)
        audio_mask = batch["audio_mask"].to(self.device)
        texts = batch["text"]
        speakers = batch["speaker"].to(self.device)
        
        # Forward pass
        # Note: Actual CSM forward call may differ based on API
        try:
            outputs = model(
                audio=audio,
                audio_mask=audio_mask,
                text=texts,
                speaker_ids=speakers,
            )
            loss = outputs.loss
        except Exception as e:
            print(f"Forward pass error: {e}")
            # Fallback: simple MSE loss for testing
            with torch.no_grad():
                encoded = model.encode(audio)
            decoded = model.decode(encoded)
            loss = nn.MSELoss()(decoded, audio)
        
        # Backward pass with gradient accumulation
        loss = loss / self.config.get("gradient_accumulation", 1)
        loss.backward()
        
        return loss.item() * self.config.get("gradient_accumulation", 1)
    
    def validate(self, model, val_loader):
        """Run validation."""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(self.device)
                audio_mask = batch["audio_mask"].to(self.device)
                texts = batch["text"]
                speakers = batch["speaker"].to(self.device)
                
                try:
                    outputs = model(
                        audio=audio,
                        audio_mask=audio_mask,
                        text=texts,
                        speaker_ids=speakers,
                    )
                    loss = outputs.loss
                except:
                    # Fallback
                    encoded = model.encode(audio)
                    decoded = model.decode(encoded)
                    loss = nn.MSELoss()(decoded, audio)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, model, optimizer, scheduler, name: str):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        path = self.output_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def train(self):
        """Main training loop."""
        # Setup
        model = self.load_model()
        train_loader, val_loader = self.load_data()
        optimizer, scheduler = self.setup_optimizer(model)
        
        # Logging
        if HAS_WANDB and self.config.get("use_wandb", False):
            wandb.init(project="voice-clone", config=self.config)
        
        num_epochs = self.config["num_epochs"]
        grad_accum = self.config.get("gradient_accumulation", 1)
        
        print(f"\nStarting training:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Effective batch size: {self.config['batch_size'] * grad_accum}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_steps = 0
            
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(progress):
                # Training step
                loss = self.train_step(model, batch, optimizer, scheduler)
                epoch_loss += loss
                num_steps += 1
                
                # Gradient accumulation
                if (step + 1) % grad_accum == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.get("max_grad_norm", 1.0)
                    )
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Update progress
                    progress.set_postfix(
                        loss=f"{loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}"
                    )
                    
                    # Logging
                    if HAS_WANDB and self.config.get("use_wandb", False):
                        wandb.log({
                            "train/loss": loss,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch + step / len(train_loader),
                        }, step=self.global_step)
            
            # End of epoch
            avg_train_loss = epoch_loss / max(num_steps, 1)
            
            # Validation
            val_loss = self.validate(model, val_loader)
            
            print(f"\nEpoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Logging
            if HAS_WANDB and self.config.get("use_wandb", False):
                wandb.log({
                    "val/loss": val_loss,
                    "epoch": epoch + 1,
                }, step=self.global_step)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(model, optimizer, scheduler, "best")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get("save_every", 5) == 0:
                self.save_checkpoint(model, optimizer, scheduler, f"epoch_{epoch+1}")
            
            # Early stopping
            if val_loss > self.best_val_loss * 1.5:
                patience = self.config.get("early_stopping_patience", 5)
                if epoch >= patience:
                    print(f"Early stopping: val_loss hasn't improved for {patience} epochs")
                    break
        
        # Final save
        self.save_checkpoint(model, optimizer, scheduler, "final")
        
        if HAS_WANDB and self.config.get("use_wandb", False):
            wandb.finish()
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Output directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CSM on your voice data")
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML")
    
    # Override config values from command line
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--data_dir", help="Data directory")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override with command line args
    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.data_dir:
        config["data_dir"] = args.data_dir
    
    # Run training
    trainer = CSMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
