"""
Training Script for ProsodyFlow

Trains a conditional flow matching model for prosody generation.
The model learns to map Gaussian noise to prosody latent vectors,
conditioned on text embeddings.

Usage:
    python train_prosody_flow.py --config config/prosody_flow.yaml

    # With custom settings
    python train_prosody_flow.py \
        --config config/prosody_flow.yaml \
        --batch_size 16 \
        --lr 1e-4 \
        --epochs 100
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from prosody_flow import (
    ProsodyFlowConfig,
    ProsodyFlow,
    ProsodyFlowAdapter,
    ProsodyFlowLoss,
)
from prosody_conditioning import (
    ProsodyConfig,
    ProsodyEncoder,
    extract_prosody_for_conditioning,
)


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger("prosody_flow")
    logger.setLevel(logging.INFO)

    # File handler
    log_file = Path(output_dir) / "training.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# =============================================================================
# DATASET
# =============================================================================

class ProsodyFlowDataset(Dataset):
    """
    Dataset for ProsodyFlow training.

    Loads prosody vectors and optional text embeddings.
    Can use pre-extracted prosody from the existing pipeline or
    extract on-the-fly from audio.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        prosody_dim: int = 2048,
        text_dim: int = 768,
        max_text_len: int = 128,
        use_cached_embeddings: bool = True,
    ):
        """
        Args:
            data_dir: Path to data directory
            split: train/val/test
            prosody_dim: Dimension of prosody vectors
            text_dim: Dimension of text embeddings
            max_text_len: Maximum text sequence length
            use_cached_embeddings: Use pre-cached embeddings if available
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.prosody_dim = prosody_dim
        self.text_dim = text_dim
        self.max_text_len = max_text_len
        self.use_cached = use_cached_embeddings

        # Load sample manifest
        manifest_path = self.data_dir / f"{split}.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.samples = json.load(f)
        else:
            # Fallback: scan directory for samples
            self.samples = self._scan_directory()

        # Initialize prosody encoder for extraction if needed
        self.prosody_config = ProsodyConfig()
        self.prosody_encoder = None  # Lazy init

    def _scan_directory(self) -> List[Dict]:
        """Scan directory for audio/prosody files."""
        samples = []
        split_dir = self.data_dir / self.split

        if not split_dir.exists():
            split_dir = self.data_dir

        for audio_path in split_dir.glob("**/*.wav"):
            sample = {
                "audio_path": str(audio_path),
                "id": audio_path.stem,
            }

            # Check for prosody file
            prosody_path = audio_path.with_suffix(".prosody.pt")
            if prosody_path.exists():
                sample["prosody_path"] = str(prosody_path)

            # Check for text embedding
            text_path = audio_path.with_suffix(".text_emb.pt")
            if text_path.exists():
                sample["text_emb_path"] = str(text_path)

            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_prosody_encoder(self) -> ProsodyEncoder:
        """Lazy initialization of prosody encoder."""
        if self.prosody_encoder is None:
            self.prosody_encoder = ProsodyEncoder(self.prosody_config)
        return self.prosody_encoder

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dict with:
                - 'prosody': [prosody_dim] target prosody vector
                - 'text_emb': [seq_len, text_dim] text embeddings (optional)
                - 'text_mask': [seq_len] attention mask (optional)
        """
        sample = self.samples[idx]
        result = {}

        # Load prosody
        if "prosody_path" in sample and Path(sample["prosody_path"]).exists():
            # Load pre-extracted prosody
            prosody = torch.load(sample["prosody_path"])
            if isinstance(prosody, dict):
                # If stored as dict of components, combine them
                prosody = self._combine_prosody_dict(prosody)
        else:
            # Generate random prosody for testing (replace with real extraction)
            prosody = torch.randn(self.prosody_dim)

        result["prosody"] = prosody

        # Load text embeddings
        if "text_emb_path" in sample and Path(sample["text_emb_path"]).exists():
            text_emb = torch.load(sample["text_emb_path"])
            seq_len = min(text_emb.shape[0], self.max_text_len)
            text_emb = text_emb[:seq_len]

            # Pad if necessary
            if text_emb.shape[0] < self.max_text_len:
                padding = torch.zeros(
                    self.max_text_len - text_emb.shape[0],
                    text_emb.shape[1]
                )
                text_emb = torch.cat([text_emb, padding], dim=0)

            # Create mask
            text_mask = torch.zeros(self.max_text_len, dtype=torch.bool)
            text_mask[:seq_len] = True

            result["text_emb"] = text_emb
            result["text_mask"] = text_mask
        else:
            # Generate dummy text embeddings for testing
            result["text_emb"] = torch.randn(self.max_text_len, self.text_dim)
            result["text_mask"] = torch.ones(self.max_text_len, dtype=torch.bool)

        return result

    def _combine_prosody_dict(self, prosody_dict: Dict) -> torch.Tensor:
        """Combine prosody dict to single vector."""
        encoder = self._get_prosody_encoder()

        semantic = prosody_dict.get("semantic", torch.zeros(8))
        acoustic = prosody_dict.get("acoustic", torch.zeros(12))
        rhythm = prosody_dict.get("rhythm", torch.zeros(8))
        contour = prosody_dict.get("contour", torch.zeros(64))

        # Use encoder to get combined representation
        with torch.no_grad():
            prefix = encoder(
                semantic.unsqueeze(0),
                acoustic.unsqueeze(0),
                rhythm.unsqueeze(0),
                contour.unsqueeze(0),
            )
            # Flatten prefix tokens
            prosody = prefix.view(-1)

        return prosody


# =============================================================================
# TRAINING
# =============================================================================

class ProsodyFlowTrainer:
    """
    Trainer for ProsodyFlow model.

    Handles:
    - Data loading
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        config: Dict,
        output_dir: str,
        device: str = "cuda",
    ):
        """
        Args:
            config: Training configuration
            output_dir: Output directory for checkpoints and logs
            device: Training device
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(str(self.output_dir))
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Config: {json.dumps(config, indent=2)}")

        # Initialize model
        self._init_model()

        # Initialize data
        self._init_data()

        # Initialize optimizer and scheduler
        self._init_optimizer()

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _init_model(self):
        """Initialize ProsodyFlow model."""
        flow_config = ProsodyFlowConfig(
            prosody_dim=self.config.get("prosody_dim", 2048),
            text_dim=self.config.get("text_dim", 768),
            use_text_conditioning=self.config.get("use_text_conditioning", True),
            sigma_min=self.config.get("sigma_min", 0.001),
            use_ot_coupling=self.config.get("use_ot_coupling", True),
            ot_reg=self.config.get("ot_reg", 0.05),
            variance_scale=self.config.get("variance_scale", 1.0),
            hidden_dim=self.config.get("hidden_dim", 512),
            num_layers=self.config.get("num_layers", 4),
            num_heads=self.config.get("num_heads", 8),
            dropout=self.config.get("dropout", 0.1),
            num_prosody_tokens=self.config.get("num_prosody_tokens", 4),
            num_ode_steps_inference=self.config.get("num_ode_steps", 50),
            ode_method=self.config.get("ode_method", "euler"),
        )

        self.model = ProsodyFlow(flow_config).to(self.device)
        self.loss_fn = ProsodyFlowLoss(
            cfm_weight=self.config.get("cfm_weight", 1.0),
            reconstruction_weight=self.config.get("reconstruction_weight", 0.0),
            kl_weight=self.config.get("kl_weight", 0.0),
        )

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model parameters: {num_params:,}")

    def _init_data(self):
        """Initialize data loaders."""
        data_dir = self.config.get("data_dir", "data/splits")

        # Training dataset
        train_dataset = ProsodyFlowDataset(
            data_dir=data_dir,
            split="train",
            prosody_dim=self.config.get("prosody_dim", 2048),
            text_dim=self.config.get("text_dim", 768),
            max_text_len=self.config.get("max_text_len", 128),
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 16),
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

        # Validation dataset
        val_dataset = ProsodyFlowDataset(
            data_dir=data_dir,
            split="val",
            prosody_dim=self.config.get("prosody_dim", 2048),
            text_dim=self.config.get("text_dim", 768),
            max_text_len=self.config.get("max_text_len", 128),
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get("batch_size", 16),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
        )

        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")

    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("lr", 1e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler
        total_steps = (
            len(self.train_loader) *
            self.config.get("epochs", 100)
        )
        warmup_steps = self.config.get("warmup_steps", 1000)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        import math
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Move to device
            prosody = batch["prosody"].to(self.device)
            text_emb = batch.get("text_emb")
            text_mask = batch.get("text_mask")

            if text_emb is not None:
                text_emb = text_emb.to(self.device)
            if text_mask is not None:
                text_mask = text_mask.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            flow_output = self.model.compute_loss(
                prosody, text_emb, text_mask
            )

            # Compute total loss
            loss = flow_output["loss"]

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get("grad_clip", 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get("grad_clip", 1.0)
                )

            self.optimizer.step()
            self.scheduler.step()

            # Update stats
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
            })

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        total_diversity = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move to device
            prosody = batch["prosody"].to(self.device)
            text_emb = batch.get("text_emb")
            text_mask = batch.get("text_mask")

            if text_emb is not None:
                text_emb = text_emb.to(self.device)
            if text_mask is not None:
                text_mask = text_mask.to(self.device)

            # Compute CFM loss
            flow_output = self.model.compute_loss(prosody, text_emb, text_mask)
            total_loss += flow_output["loss"].item()

            # Sample and measure diversity
            samples = self.model.sample(
                text_emb, text_mask,
                num_samples=3,
                temperature=1.0,
            )
            # Reshape to [batch, num_samples, dim] for diversity calc
            samples_reshaped = samples.view(-1, 3, samples.shape[-1])
            diversity = samples_reshaped.std(dim=1).mean()
            total_diversity += diversity.item()

            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_diversity = total_diversity / max(num_batches, 1)

        return {
            "val_loss": avg_loss,
            "val_diversity": avg_diversity,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }

        # Save latest
        latest_path = self.output_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.output_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with val_loss={self.best_val_loss:.4f}")

        # Save periodic checkpoint
        if epoch % self.config.get("save_every", 10) == 0:
            epoch_path = self.output_dir / f"epoch_{epoch:04d}.pt"
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        return checkpoint["epoch"]

    def train(self):
        """Main training loop."""
        start_epoch = 0

        # Resume from checkpoint if specified
        if self.config.get("resume"):
            start_epoch = self.load_checkpoint(self.config["resume"]) + 1

        num_epochs = self.config.get("epochs", 100)

        self.logger.info(f"Starting training from epoch {start_epoch}")

        for epoch in range(start_epoch, num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['train_loss']:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"val_diversity={val_metrics['val_diversity']:.4f}"
            )

            # Save checkpoint
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            self.save_checkpoint(epoch, is_best)

            # Early stopping
            if self.config.get("early_stopping_patience"):
                # Implement if needed
                pass

        self.logger.info("Training complete!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ProsodyFlow model")

    parser.add_argument(
        "--config",
        type=str,
        default="config/prosody_flow.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        config = {}

    # Override with command line args
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["lr"] = args.lr
    if args.epochs:
        config["epochs"] = args.epochs
    if args.resume:
        config["resume"] = args.resume

    # Set default output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"checkpoints/prosody_flow_{timestamp}"

    # Create trainer
    trainer = ProsodyFlowTrainer(
        config=config,
        output_dir=output_dir,
        device=args.device,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
