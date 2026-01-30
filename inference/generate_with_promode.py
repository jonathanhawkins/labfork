#!/usr/bin/env python3
"""
ProMode Inference Script

Generate prosody predictions from text using the ProMode model.
Can be used standalone or integrated with downstream TTS systems.

Based on "ProMode: A Speech Prosody Model Conditioned on Acoustic and Textual Inputs"
(Interspeech 2025) - https://arxiv.org/abs/2508.09389

Usage:
    # Generate prosody from text
    python generate_with_promode.py \
        --text "Hello, how are you?" \
        --checkpoint ../checkpoints/promode/best.pt \
        --output prosody.pt

    # Generate with reference audio (for comparison)
    python generate_with_promode.py \
        --text "Hello, how are you?" \
        --reference reference.wav \
        --checkpoint ../checkpoints/promode/best.pt \
        --output prosody.pt

    # Visualize prosody predictions
    python generate_with_promode.py \
        --text "Hello, how are you?" \
        --checkpoint ../checkpoints/promode/best.pt \
        --visualize \
        --output prosody.png

    # Export for downstream TTS
    python generate_with_promode.py \
        --text "Hello, how are you?" \
        --checkpoint ../checkpoints/promode/best.pt \
        --export-format fluentspeech \
        --output prosody_fluentspeech.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

from promode import (
    ProModeConfig,
    ProMode,
    ProModeAdapter,
    create_promode_adapter,
)


def load_text_encoder(model_name: str = "bert-base-uncased"):
    """Load text encoder for generating text features."""
    try:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return model, tokenizer
    except ImportError:
        print("Warning: transformers not installed, using mock text features")
        return None, None


def text_to_features(
    text: str,
    model,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Convert text to features using text encoder."""
    if model is None or tokenizer is None:
        # Mock features
        text_len = len(text.split()) + 5
        return torch.randn(1, text_len, 512).to(device)

    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        outputs = model.to(device)(**inputs)
        features = outputs.last_hidden_state  # [1, seq, dim]

    return features


def visualize_prosody(
    f0: torch.Tensor,
    energy: torch.Tensor,
    vuv: torch.Tensor,
    output_path: str,
    text: str = "",
):
    """Visualize prosody predictions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping visualization")
        return

    f0_np = f0.cpu().numpy().flatten()
    energy_np = energy.cpu().numpy().flatten()
    vuv_np = vuv.cpu().numpy().flatten()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # F0
    ax1 = axes[0]
    ax1.plot(f0_np, 'b-', linewidth=1.5)
    ax1.fill_between(range(len(f0_np)), 0, f0_np, alpha=0.3)
    ax1.set_ylabel('F0 (Hz)')
    ax1.set_title(f'ProMode Prosody Prediction\n"{text[:50]}..."' if len(text) > 50 else f'ProMode Prosody Prediction\n"{text}"')
    ax1.grid(True, alpha=0.3)

    # Energy
    ax2 = axes[1]
    ax2.plot(energy_np, 'g-', linewidth=1.5)
    ax2.fill_between(range(len(energy_np)), 0, energy_np, alpha=0.3, color='green')
    ax2.set_ylabel('Energy')
    ax2.grid(True, alpha=0.3)

    # Voiced/Unvoiced
    ax3 = axes[2]
    ax3.plot(vuv_np, 'r-', linewidth=1.5)
    ax3.fill_between(range(len(vuv_np)), 0, vuv_np, alpha=0.3, color='red')
    ax3.set_ylabel('Voiced Prob')
    ax3.set_xlabel('Frame')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Visualization saved to {output_path}")


def export_for_fluentspeech(
    prosody: dict,
    output_path: str,
):
    """Export prosody in FluentSpeech-compatible format."""
    export_data = {
        'f0': prosody['f0'].cpu(),
        'energy': prosody['energy'].cpu(),
        'duration': prosody.get('duration', torch.ones_like(prosody['f0'])).cpu(),
        'vuv': prosody.get('vuv', (prosody['f0'] > 0).float()).cpu(),
    }

    torch.save(export_data, output_path)
    print(f"Exported FluentSpeech format to {output_path}")


def export_for_csm(
    prosody: dict,
    adapter: ProModeAdapter,
    text_features: torch.Tensor,
    output_path: str,
):
    """Export prosody tokens for CSM."""
    result = adapter.from_text(text_features)

    export_data = {
        'prosody_tokens': result['prosody_tokens'].cpu(),
        'f0': result.get('f0', prosody['f0']).cpu(),
        'energy': result.get('energy', prosody['energy']).cpu(),
        'latent_prosody': result.get('latent_prosody', prosody['latent_prosody']).cpu(),
    }

    torch.save(export_data, output_path)
    print(f"Exported CSM format to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate prosody with ProMode")
    parser.add_argument('--text', type=str, required=True,
                       help='Text to generate prosody for')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to ProMode checkpoint')
    parser.add_argument('--output', type=str, default='prosody.pt',
                       help='Output path')
    parser.add_argument('--reference', type=str, default=None,
                       help='Reference audio for comparison (optional)')
    parser.add_argument('--text-encoder', type=str, default='bert-base-uncased',
                       help='Text encoder model name')
    parser.add_argument('--target-length', type=int, default=None,
                       help='Target acoustic sequence length')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize prosody predictions')
    parser.add_argument('--export-format', type=str, default=None,
                       choices=['fluentspeech', 'csm', 'raw'],
                       help='Export format for downstream TTS')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, mps, cpu)')
    args = parser.parse_args()

    print("=" * 60)
    print("ProMode Inference")
    print("=" * 60)

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Load text encoder
    print(f"\nLoading text encoder: {args.text_encoder}")
    text_model, tokenizer = load_text_encoder(args.text_encoder)

    # Load ProMode
    print(f"Loading ProMode from {args.checkpoint}")
    adapter = create_promode_adapter(checkpoint=args.checkpoint)
    adapter = adapter.to(device)
    adapter.eval()

    # Generate text features
    print(f"\nText: {args.text}")
    text_features = text_to_features(args.text, text_model, tokenizer, device)
    print(f"Text features shape: {text_features.shape}")

    # Generate prosody
    print("\nGenerating prosody...")
    with torch.no_grad():
        prosody = adapter.promode.predict_prosody(
            text_features,
            target_length=args.target_length,
        )

    f0 = prosody['f0']
    energy = prosody['energy']
    vuv = prosody['vuv']
    duration = prosody['duration']
    latent = prosody['latent_prosody']

    print(f"\nProsody predictions:")
    print(f"  F0 shape: {f0.shape}")
    print(f"  F0 range: {f0.min().item():.1f} - {f0.max().item():.1f} Hz")
    print(f"  Energy shape: {energy.shape}")
    print(f"  VUV shape: {vuv.shape}")
    print(f"  Duration shape: {duration.shape}")
    print(f"  Latent shape: {latent.shape}")

    # Visualize if requested
    if args.visualize:
        vis_path = args.output.replace('.pt', '.png')
        visualize_prosody(f0, energy, vuv, vis_path, args.text)

    # Export based on format
    output_path = args.output

    if args.export_format == 'fluentspeech':
        export_for_fluentspeech(prosody, output_path)
    elif args.export_format == 'csm':
        export_for_csm(prosody, adapter, text_features, output_path)
    else:
        # Raw format
        export_data = {
            'text': args.text,
            'f0': f0.cpu(),
            'energy': energy.cpu(),
            'vuv': vuv.cpu(),
            'duration': duration.cpu(),
            'latent_prosody': latent.cpu(),
        }

        # Get prefix tokens for CSM integration
        result = adapter.from_text(text_features)
        export_data['prosody_tokens'] = result['prosody_tokens'].cpu()

        torch.save(export_data, output_path)
        print(f"\nSaved prosody to {output_path}")

    # Compare with reference if provided
    if args.reference:
        print(f"\nComparing with reference: {args.reference}")
        try:
            import torchaudio

            audio, sr = torchaudio.load(args.reference)
            audio = audio[0]  # Mono

            # Extract reference F0 (requires pyworld)
            try:
                import pyworld as pw
                import numpy as np

                audio_np = audio.numpy().astype(np.float64)
                f0_ref, t = pw.dio(audio_np, sr, frame_period=16.0)
                f0_ref = pw.stonemask(audio_np, f0_ref, t, sr)

                f0_pred = f0[0].cpu().numpy()

                # Align lengths
                min_len = min(len(f0_ref), len(f0_pred))
                f0_ref = f0_ref[:min_len]
                f0_pred = f0_pred[:min_len]

                # Compute metrics (only on voiced frames)
                voiced = f0_ref > 0
                if voiced.sum() > 0:
                    from scipy.stats import pearsonr

                    corr, _ = pearsonr(f0_ref[voiced], f0_pred[voiced])
                    rmse = np.sqrt(((f0_ref[voiced] - f0_pred[voiced]) ** 2).mean())

                    print(f"\nF0 Metrics (vs reference):")
                    print(f"  Correlation: {corr:.4f}")
                    print(f"  RMSE: {rmse:.2f} Hz")
            except ImportError:
                print("  Note: pyworld not installed, skipping F0 comparison")
        except Exception as e:
            print(f"  Error loading reference: {e}")

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
