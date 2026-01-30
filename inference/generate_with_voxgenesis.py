#!/usr/bin/env python3
"""
Inference script for VoxGenesis latent speaker manifold.

Based on "VoxGenesis: Interpretable Voice Synthesis and Manipulation"
arXiv:2403.00529 (March 2024)

Features:
- Novel speaker generation by sampling from Gaussian
- Voice editing via latent manipulation along discovered directions
- Voice conversion (source content + target speaker)
- Direction sweeps for visualization

Usage:
    # Generate with novel speaker
    python generate_with_voxgenesis.py \
        --text "Hello, how are you?" \
        --checkpoint ../checkpoints/voxgenesis/best.pt \
        --novel-speaker \
        --output novel_speaker.wav

    # Voice editing (e.g., increase pitch)
    python generate_with_voxgenesis.py \
        --text "Hello, how are you?" \
        --reference speaker.wav \
        --edit pitch:0.5 \
        --checkpoint ../checkpoints/voxgenesis/best.pt \
        --output edited.wav

    # Voice conversion
    python generate_with_voxgenesis.py \
        --text "Hello, how are you?" \
        --source content.wav \
        --target speaker.wav \
        --checkpoint ../checkpoints/voxgenesis/best.pt \
        --output converted.wav

    # Sweep a direction (generate multiple scales)
    python generate_with_voxgenesis.py \
        --text "Hello, how are you?" \
        --reference speaker.wav \
        --sweep pitch \
        --checkpoint ../checkpoints/voxgenesis/best.pt \
        --output outputs/

    # Generate multiple novel speakers
    python generate_with_voxgenesis.py \
        --text "Hello, how are you?" \
        --novel-speaker --num-speakers 5 \
        --checkpoint ../checkpoints/voxgenesis/best.pt \
        --output outputs/

    # List available directions
    python generate_with_voxgenesis.py \
        --checkpoint ../checkpoints/voxgenesis/best.pt \
        --list-directions
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from voxgenesis import (
    VoxGenesisConfig,
    VoxGenesis,
    VoxGenesisAdapter,
    create_voxgenesis_adapter,
)


def load_model(
    checkpoint_path: str,
    directions_path: Optional[str] = None,
    device: str = "cuda",
) -> VoxGenesis:
    """Load VoxGenesis model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config if present
    if 'config' in checkpoint:
        config_dict = checkpoint['config'].get('model', {})
        config = VoxGenesisConfig(
            input_dim=config_dict.get('input_dim', 768),
            semantic_dim=config_dict.get('semantic_dim', 768),
            mel_dim=config_dict.get('mel_dim', 80),
            latent_dim=config_dict.get('latent_dim', 512),
            mapping_dim=config_dict.get('mapping_dim', 512),
            num_mapping_layers=config_dict.get('num_mapping_layers', 7),
            num_directions=config_dict.get('num_directions', 32),
            output_dim=config_dict.get('output_dim', 2048),
            num_prefix_tokens=config_dict.get('num_prefix_tokens', 4),
        )
    else:
        config = VoxGenesisConfig()

    model = VoxGenesis(config)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Load directions if available
    if directions_path is None:
        # Try to find directions in same directory as checkpoint
        checkpoint_dir = Path(checkpoint_path).parent
        default_directions = checkpoint_dir / "directions.pt"
        if default_directions.exists():
            directions_path = str(default_directions)

    if directions_path and os.path.exists(directions_path):
        directions_data = torch.load(directions_path, map_location=device)
        model.direction_discovery.directions.copy_(directions_data['directions'])
        model.direction_discovery.eigenvalues.copy_(directions_data['eigenvalues'])
        model.direction_discovery.mean.copy_(directions_data['mean'])
        model.direction_discovery.discovered.fill_(True)

        if 'labels' in directions_data:
            model.direction_discovery.direction_labels = directions_data['labels']

        print(f"Loaded {len(directions_data['directions'])} directions from {directions_path}")
        print(f"Available labels: {list(model.direction_discovery.direction_labels.keys())}")

    return model


def extract_features(
    audio_path: str,
    model_name: str = "facebook/hubert-large-ls960-ft",
    device: str = "cuda",
) -> torch.Tensor:
    """Extract HuBERT features from audio file."""
    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Try to use HuBERT, fall back to synthetic
    try:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = HubertModel.from_pretrained(model_name).to(device)
        model.eval()

        inputs = feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(inputs.input_values.to(device))
            features = outputs.last_hidden_state  # [1, seq, 768]

        return features
    except Exception as e:
        print(f"Warning: Could not load HuBERT ({e}), using synthetic features")
        # Return synthetic features based on audio length
        num_frames = waveform.shape[-1] // 320
        return torch.randn(1, num_frames, 768).to(device)


def save_audio(mel: torch.Tensor, output_path: str, sample_rate: int = 24000):
    """Convert mel spectrogram to audio and save."""
    # In a real implementation, use a vocoder (e.g., HiFi-GAN)
    # For now, just save the mel spectrogram as a placeholder
    print(f"Generated mel spectrogram shape: {mel.shape}")
    print(f"Would save audio to: {output_path}")

    # Create a placeholder audio file (silence with correct length)
    # In production, replace with vocoder
    duration_sec = mel.shape[1] / 100  # Approximate
    num_samples = int(duration_sec * sample_rate)
    audio = torch.zeros(1, num_samples)
    torchaudio.save(output_path, audio, sample_rate)
    print(f"Saved placeholder audio to {output_path}")


def generate_novel_speaker(
    model: VoxGenesis,
    semantic_tokens: torch.Tensor,
    num_speakers: int = 1,
    temperature: float = 1.0,
    truncation_psi: float = 1.0,
) -> List[Dict[str, torch.Tensor]]:
    """Generate speech with novel speakers."""
    device = semantic_tokens.device
    batch_size = semantic_tokens.shape[0]

    results = []
    for i in range(num_speakers):
        # Sample from Gaussian
        z = torch.randn(batch_size, model.config.latent_dim, device=device) * temperature

        # Apply truncation trick if specified
        if truncation_psi < 1.0:
            # Move z towards mean (0)
            z = z * truncation_psi

        # Generate
        gen_output = model.generate_from_z(z, semantic_tokens)
        results.append({
            'mel': gen_output['mel'],
            'z': z,
            'w': gen_output['w'],
        })

    return results


def voice_editing(
    model: VoxGenesis,
    speaker_features: torch.Tensor,
    semantic_tokens: torch.Tensor,
    edits: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """Apply voice editing via latent manipulation."""
    # Encode speaker
    encoder_output = model.encode_speaker(speaker_features)
    z = encoder_output['z']
    w = model.mapping_network(z)

    # Apply edits
    w_edited = w.clone()
    for direction_name, scale in edits.items():
        if direction_name in model.direction_discovery.direction_labels:
            direction_idx = model.direction_discovery.direction_labels[direction_name]
            w_edited = model.direction_discovery.manipulate(w_edited, direction_idx, scale)
        elif direction_name.isdigit():
            direction_idx = int(direction_name)
            w_edited = model.direction_discovery.manipulate(w_edited, direction_idx, scale)
        else:
            print(f"Warning: Unknown direction '{direction_name}'")

    # Generate with edited w
    semantic_output = model.semantic_embedding(semantic_tokens)
    semantic_emb = semantic_output['embedding']
    transformed = model.transformation(w_edited, semantic_emb)
    conditioned = transformed + semantic_emb
    mel_edited = model.deconvolution(conditioned)

    return {
        'original_w': w,
        'edited_w': w_edited,
        'mel': mel_edited,
    }


def direction_sweep(
    model: VoxGenesis,
    speaker_features: torch.Tensor,
    semantic_tokens: torch.Tensor,
    direction: str,
    scales: List[float] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
) -> List[Dict[str, torch.Tensor]]:
    """Sweep across a direction to visualize its effect."""
    results = []

    for scale in scales:
        edit_output = voice_editing(
            model, speaker_features, semantic_tokens,
            edits={direction: scale}
        )
        results.append({
            'scale': scale,
            'mel': edit_output['mel'],
        })

    return results


def list_directions(model: VoxGenesis):
    """Print available directions."""
    print("\nAvailable Directions:")
    print("-" * 50)

    if not model.direction_discovery.discovered:
        print("No directions discovered yet. Run direction discovery first.")
        return

    info = model.direction_discovery.get_direction_info()

    for i in range(min(15, model.config.num_directions)):
        var = info['explained_variance_ratio'][i] * 100
        label = ""
        for name, idx in info['labels'].items():
            if idx == i:
                label = f" [{name}]"
                break
        print(f"  {i:2d}: {var:5.2f}% variance{label}")

    print(f"\nLabeled directions: {list(info['labels'].keys())}")
    print(f"Cumulative variance (top 5): {info['cumulative_variance'][4]*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Generate audio with VoxGenesis")

    # Input/Output
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--reference", type=str, help="Reference speaker audio")
    parser.add_argument("--source", type=str, help="Source audio for content (voice conversion)")
    parser.add_argument("--target", type=str, help="Target speaker audio (voice conversion)")
    parser.add_argument("--output", type=str, required=True, help="Output path")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--directions", type=str, default=None,
                        help="Path to directions file (optional)")

    # Generation modes
    parser.add_argument("--novel-speaker", action="store_true",
                        help="Generate with novel speaker")
    parser.add_argument("--num-speakers", type=int, default=1,
                        help="Number of novel speakers to generate")
    parser.add_argument("--edit", type=str, action="append", default=[],
                        help="Voice edit: direction:scale (e.g., 'pitch:0.5')")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Sweep a direction")
    parser.add_argument("--list-directions", action="store_true",
                        help="List available directions")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for novel speaker sampling")
    parser.add_argument("--truncation", type=float, default=1.0,
                        help="Truncation psi for novel speakers (0=mean, 1=full)")

    args = parser.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.directions, device)

    # List directions mode
    if args.list_directions:
        list_directions(model)
        return

    # We need either text or source audio
    if not args.text and not args.source:
        print("Error: Either --text or --source is required")
        return

    # Get semantic tokens (from text or source audio)
    if args.source:
        print(f"Extracting features from source: {args.source}")
        semantic_tokens = extract_features(args.source, device=device)
    else:
        # In a real implementation, convert text to semantic tokens
        # For now, use synthetic
        print("Using synthetic semantic tokens (text-to-speech not implemented)")
        semantic_tokens = torch.randn(1, 100, model.config.semantic_dim).to(device)

    # Generate based on mode
    output_path = Path(args.output)

    # Novel speaker generation
    if args.novel_speaker:
        print(f"\nGenerating {args.num_speakers} novel speaker(s)...")
        results = generate_novel_speaker(
            model, semantic_tokens,
            num_speakers=args.num_speakers,
            temperature=args.temperature,
            truncation_psi=args.truncation,
        )

        if args.num_speakers == 1:
            save_audio(results[0]['mel'], str(output_path))
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            for i, result in enumerate(results):
                out_file = output_path / f"novel_speaker_{i}.wav"
                save_audio(result['mel'], str(out_file))

        print("Done!")
        return

    # Voice conversion
    if args.target:
        if not args.source:
            print("Error: --source required for voice conversion")
            return

        print(f"Performing voice conversion...")
        print(f"  Content from: {args.source}")
        print(f"  Speaker from: {args.target}")

        target_features = extract_features(args.target, device=device)
        gen_output = model.voice_conversion(semantic_tokens, target_features)
        save_audio(gen_output['mel'], str(output_path))
        print("Done!")
        return

    # Voice editing
    if args.edit or args.sweep:
        if not args.reference:
            print("Error: --reference required for voice editing")
            return

        print(f"Extracting reference speaker from: {args.reference}")
        speaker_features = extract_features(args.reference, device=device)

        # Direction sweep
        if args.sweep:
            print(f"\nSweeping direction: {args.sweep}")
            results = direction_sweep(
                model, speaker_features, semantic_tokens,
                direction=args.sweep,
            )

            output_path.mkdir(parents=True, exist_ok=True)
            for result in results:
                scale_str = f"{result['scale']:.1f}".replace("-", "neg")
                out_file = output_path / f"{args.sweep}_scale_{scale_str}.wav"
                save_audio(result['mel'], str(out_file))

            print("Done!")
            return

        # Single edit(s)
        if args.edit:
            edits = {}
            for edit_str in args.edit:
                parts = edit_str.split(":")
                if len(parts) != 2:
                    print(f"Warning: Invalid edit format '{edit_str}' (expected 'direction:scale')")
                    continue
                direction, scale = parts[0], float(parts[1])
                edits[direction] = scale

            print(f"Applying edits: {edits}")
            edit_output = voice_editing(model, speaker_features, semantic_tokens, edits)
            save_audio(edit_output['mel'], str(output_path))
            print("Done!")
            return

    # Default: just generate with reference speaker
    if args.reference:
        print(f"Generating with reference speaker from: {args.reference}")
        speaker_features = extract_features(args.reference, device=device)
        gen_output = model.generate_from_features(speaker_features, semantic_tokens)
        save_audio(gen_output['mel'], str(output_path))
        print("Done!")
        return

    print("Error: Specify --reference, --novel-speaker, or --target for voice conversion")


if __name__ == "__main__":
    main()
