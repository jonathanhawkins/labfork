"""
CSM Fine-Tuning Script - Uses processor with output_labels=True
Optimized for RTX 4090 and CSM model from HuggingFace
With Dashboard API support for real-time monitoring
"""

import argparse
import json
import torch
import torchaudio
import threading
import queue
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import yaml
import sys


# ============== Metrics for Dashboard ==============

@dataclass
class TrainingMetrics:
    """Real-time training metrics for dashboard."""
    step: int = 0
    epoch: int = 0
    epoch_progress: float = 0.0
    total_epochs: int = 50

    # Losses
    train_loss: float = 0.0
    val_loss: float = 0.0

    # Learning rate
    learning_rate: float = 0.0

    # Performance
    samples_per_second: float = 0.0
    batch_time: float = 0.0

    # Memory
    memory_used_gb: float = 0.0
    memory_peak_gb: float = 0.0

    # History for charts
    loss_history: List[Dict[str, float]] = field(default_factory=list)
    lr_history: List[Dict[str, float]] = field(default_factory=list)

    # Status
    status: str = "initializing"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsLogger:
    """Thread-safe metrics logger with WebSocket support."""

    def __init__(self, max_history: int = 1000):
        self.metrics = TrainingMetrics()
        self.max_history = max_history
        self.lock = threading.Lock()
        self.subscribers: List[queue.Queue] = []

    def update(self, **kwargs):
        """Update metrics and notify subscribers."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)

            # Add to history
            if 'train_loss' in kwargs:
                self.metrics.loss_history.append({
                    'step': self.metrics.step,
                    'epoch': self.metrics.epoch,
                    'train_loss': self.metrics.train_loss,
                    'val_loss': self.metrics.val_loss,
                })
                if len(self.metrics.loss_history) > self.max_history:
                    self.metrics.loss_history.pop(0)

            # Notify subscribers
            data = self.metrics.to_dict()
            for q in self.subscribers:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=100)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            return self.metrics.to_dict()


# ============== Dataset ==============

class CSMDataset(Dataset):
    """Dataset that uses processor with output_labels=True."""

    def __init__(self, data_path, processor, max_audio_length_ms=30000, sample_rate=24000):
        with open(data_path) as f:
            self.samples = json.load(f)
        self.processor = processor
        self.max_audio_length = int(max_audio_length_ms * sample_rate / 1000)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        audio_path = sample.get('audio_path') or sample.get('path')
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] > self.max_audio_length:
            waveform = waveform[:, :self.max_audio_length]

        audio = waveform.squeeze(0).numpy()
        text = sample['text']
        speaker = str(sample.get('speaker', 0))

        # Build conversation with audio
        conversation = [{
            'role': speaker,
            'content': [
                {'type': 'text', 'text': text},
                {'type': 'audio', 'audio': audio}
            ]
        }]

        # Use processor with output_labels=True
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, output_labels=True
        )

        inputs['seq_len'] = inputs['input_ids'].shape[-1]
        inputs['audio_len'] = len(audio)

        return inputs


def collate_fn(batch):
    """Collate with padding."""
    max_seq_len = max(b['seq_len'] for b in batch)
    max_audio_len = max(b['audio_len'] for b in batch)
    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
    input_values = torch.zeros(batch_size, 1, max_audio_len, dtype=torch.float32)
    input_values_cutoffs = torch.zeros(batch_size, 1, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b['seq_len']
        audio_len = b['audio_len']

        input_ids[i, :seq_len] = b['input_ids'].squeeze(0)
        attention_mask[i, :seq_len] = b['attention_mask'].squeeze(0)
        labels[i, :seq_len] = b['labels'].squeeze(0)
        input_values[i, :, :audio_len] = b['input_values'].squeeze(0)
        input_values_cutoffs[i] = b['input_values_cutoffs'].squeeze()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'input_values': input_values,
        'input_values_cutoffs': input_values_cutoffs,
    }


# ============== Dashboard API ==============

def create_dashboard_api(logger: MetricsLogger):
    """Create FastAPI server for training dashboard."""
    from fastapi import FastAPI, WebSocket
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="CSM Training Dashboard API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/metrics")
    async def get_metrics():
        return logger.get_metrics()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        q = logger.subscribe()
        try:
            while True:
                try:
                    data = q.get(timeout=1)
                    await websocket.send_json(data)
                except queue.Empty:
                    # Send heartbeat
                    await websocket.send_json(logger.get_metrics())
        except Exception:
            pass
        finally:
            logger.unsubscribe(q)

    return app


# ============== Training ==============

def train(config, logger: MetricsLogger = None):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    sys.stdout.flush()

    if logger:
        logger.update(status="loading_model")

    # Load model
    from transformers import CsmForConditionalGeneration, AutoProcessor

    model_path = config['model_path']
    print(f'Loading model from {model_path}')
    sys.stdout.flush()

    model = CsmForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float32,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print('Gradient checkpointing enabled')
    elif hasattr(model.backbone, 'gradient_checkpointing_enable'):
        model.backbone.gradient_checkpointing_enable()
        print('Backbone gradient checkpointing enabled')
    sys.stdout.flush()

    # Freeze codec
    model.codec_model.eval()
    for param in model.codec_model.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable:,}')
    sys.stdout.flush()

    # Load data
    data_dir = config['data_dir']
    train_dataset = CSMDataset(
        f'{data_dir}/train.json', processor, config.get('max_audio_length_ms', 30000)
    )
    val_dataset = CSMDataset(
        f'{data_dir}/val.json', processor, config.get('max_audio_length_ms', 30000)
    )

    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)} samples')
    sys.stdout.flush()

    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 0)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )

    # Optimizer
    learning_rate = config.get('learning_rate', 1e-5)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=config.get('weight_decay', 0.01),
        )
        print('Using 8-bit AdamW optimizer')
    except ImportError:
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=config.get('weight_decay', 0.01),
        )
        print('Using standard AdamW optimizer')
    sys.stdout.flush()

    # Training setup
    num_epochs = config.get('num_epochs', 10)
    save_dir = Path(config.get('output_dir', '../models/checkpoints/csm_final'))
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    global_step = 0

    if logger:
        logger.update(
            total_epochs=num_epochs,
            learning_rate=learning_rate,
            status="warming_up"
        )

    print(f'\nStarting training for {num_epochs} epochs...')
    print(f'Batch size: {batch_size}, Steps/epoch: {len(train_loader)}')
    sys.stdout.flush()

    # Warmup pass
    print('Running warmup pass...')
    sys.stdout.flush()

    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            try:
                _ = model(**batch)
                print('Warmup complete!')
                sys.stdout.flush()
            except Exception as e:
                print(f'Warmup error (may be expected): {e}')
                sys.stdout.flush()
            break

    # Training loop
    for epoch in range(num_epochs):
        if logger:
            logger.update(epoch=epoch + 1, status="training")

        model.train()
        model.codec_model.eval()
        total_train_loss = 0
        num_batches = 0
        epoch_start_time = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()

            try:
                outputs = model(**batch)
                loss = outputs.loss

                if loss is None:
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0))
                optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1
                global_step += 1

                batch_time = time.time() - batch_start_time
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                # Update metrics
                if logger:
                    mem_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    mem_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    logger.update(
                        step=global_step,
                        epoch=epoch + 1,
                        epoch_progress=(batch_idx + 1) / len(train_loader),
                        train_loss=loss.item(),
                        batch_time=batch_time,
                        samples_per_second=batch_size / batch_time,
                        memory_used_gb=mem_used,
                        memory_peak_gb=mem_peak,
                    )

            except Exception as e:
                print(f'Train error batch {batch_idx}: {e}')
                sys.stdout.flush()
                continue

        avg_train_loss = total_train_loss / max(num_batches, 1)

        # Validate
        if logger:
            logger.update(status="validating")

        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                try:
                    outputs = model(**batch)
                    if outputs.loss is not None:
                        total_val_loss += outputs.loss.item()
                        num_val_batches += 1
                except Exception as e:
                    pass

        avg_val_loss = total_val_loss / max(num_val_batches, 1)

        print(f'Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}')
        sys.stdout.flush()

        if logger:
            logger.update(
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                status="saving" if avg_val_loss < best_val_loss else "training"
            )

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_val_loss,
            }, save_dir / 'best.pt')
            print(f'  Saved best model (val_loss={avg_val_loss:.4f})')
            sys.stdout.flush()

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pt')

    if logger:
        logger.update(status="complete")

    print(f'\nTraining complete! Best val_loss: {best_val_loss:.4f}')
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/rtx_4090_deepseek.yaml')
    parser.add_argument('--dashboard', action='store_true', help='Start dashboard API on port 8001')
    parser.add_argument('--dashboard_port', type=int, default=8001)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create metrics logger
    logger = MetricsLogger()

    # Start dashboard if requested
    if args.dashboard:
        import uvicorn

        api = create_dashboard_api(logger)

        def run_api():
            uvicorn.run(api, host="0.0.0.0", port=args.dashboard_port, log_level="warning")

        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        print(f'Dashboard API running on http://0.0.0.0:{args.dashboard_port}')
        print(f'  WebSocket: ws://localhost:{args.dashboard_port}/ws')
        print(f'  Metrics: http://localhost:{args.dashboard_port}/metrics')
        sys.stdout.flush()
        time.sleep(1)  # Let API start

    # Run training
    train(config, logger if args.dashboard else None)


if __name__ == '__main__':
    main()
