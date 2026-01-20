"""
CSM Fine-Tuning Script - Uses processor with output_labels=True
Optimized for RTX 4090 and CSM model from HuggingFace
"""

import argparse
import json
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import yaml
import sys


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

        # Store sequence length for collation
        inputs['seq_len'] = inputs['input_ids'].shape[-1]
        inputs['audio_len'] = len(audio)

        return inputs


def collate_fn(batch):
    """Collate with padding."""
    max_seq_len = max(b['seq_len'] for b in batch)
    max_audio_len = max(b['audio_len'] for b in batch)
    batch_size = len(batch)

    # Pad tensors
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/rtx_4090_deepseek.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    sys.stdout.flush()

    # Load model
    from transformers import CsmForConditionalGeneration, AutoProcessor

    model_path = config['model_path']
    print(f'Loading model from {model_path}')
    sys.stdout.flush()

    model = CsmForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

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
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.get('learning_rate', 1e-5),
        weight_decay=config.get('weight_decay', 0.01),
    )

    # Training loop
    num_epochs = config.get('num_epochs', 10)
    save_dir = Path(config.get('output_dir', '../models/checkpoints/csm_final'))
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    print(f'\nStarting training for {num_epochs} epochs...')
    print(f'Batch size: {batch_size}, Steps/epoch: {len(train_loader)}')
    sys.stdout.flush()

    # Warmup forward pass to compile any JIT code
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

    for epoch in range(num_epochs):
        # Train
        model.train()
        model.codec_model.eval()  # Keep codec frozen
        total_train_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()

            try:
                outputs = model(**batch)
                loss = outputs.loss

                if loss is None:
                    print(f'Batch {batch_idx}: loss is None, skipping')
                    sys.stdout.flush()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0))
                optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            except Exception as e:
                print(f'Train error batch {batch_idx}: {e}')
                sys.stdout.flush()
                import traceback
                traceback.print_exc()
                continue

        avg_train_loss = total_train_loss / max(num_batches, 1)

        # Validate
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
                    print(f'Val error: {e}')
                    sys.stdout.flush()

        avg_val_loss = total_val_loss / max(num_val_batches, 1)

        print(f'Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}')
        sys.stdout.flush()

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

    print(f'\nTraining complete! Best val_loss: {best_val_loss:.4f}')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
