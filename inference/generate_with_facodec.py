#!/usr/bin/env python3
"""
Generate Speech with FACodec Prosody Conditioning

Uses NaturalSpeech3's FACodec to extract clean prosody from a reference audio
and conditions speech generation on these disentangled prosody codes.

This enables:
1. Prosody transfer: Copy prosody from reference, generate with different text
2. Clean conditioning: No speaker/content leakage in prosody representation
3. Prosody manipulation: Modify prosody codes before generation

Usage:
    # Basic generation with prosody from reference
    python generate_with_facodec.py \
        --reference reference.wav \
        --text "Hello, how are you?" \
        --output output.wav

    # Prosody transfer (reference prosody + different speaker)
    python generate_with_facodec.py \
        --reference prosody_source.wav \
        --speaker speaker_reference.wav \
        --text "New text with transferred prosody" \
        --output transferred.wav

    # Extract and save prosody codes
    python generate_with_facodec.py \
        --reference audio.wav \
        --extract-only \
        --output prosody_codes.pt

    # Generate from saved prosody codes
    python generate_with_facodec.py \
        --prosody-file prosody_codes.pt \
        --text "Text to synthesize" \
        --output output.wav
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torchaudio

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.facodec_integration import (
    FACodecConfig,
    FACodecProsodyExtractor,
    FACodecProsodyAdapter,
    FACodecControlledCSM,
    extract_facodec_prosody,
)


def extract_prosody_features(
    audio_path: str,
    extractor: FACodecProsodyExtractor,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Extract FACodec prosody features from audio.

    Args:
        audio_path: Path to audio file
        extractor: Pre-loaded FACodec extractor
        device: Target device

    Returns:
        Dict with prosody_codes, prosody_emb, speaker_emb
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Resample to 16kHz
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Add batch dimension
    waveform = waveform.unsqueeze(0).to(device)  # [1, 1, samples]

    # Extract features
    with torch.no_grad():
        result = extractor(waveform, return_all=True)

    return result


def generate_with_facodec_prosody(
    text: str,
    prosody_codes: torch.Tensor,
    speaker_emb: Optional[torch.Tensor] = None,
    model: Optional[FACodecControlledCSM] = None,
    tokenizer=None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate speech conditioned on FACodec prosody codes.

    Args:
        text: Text to synthesize
        prosody_codes: [1, time] FACodec prosody codes
        speaker_emb: [1, 256] speaker embedding (optional)
        model: Pre-loaded FACodec-conditioned model
        tokenizer: Text tokenizer
        device: Target device

    Returns:
        [1, samples] generated audio
    """
    if model is None:
        raise ValueError("Model must be provided")

    # Tokenize text
    if tokenizer is None:
        # Mock tokenization for demo
        print("Warning: No tokenizer provided, using mock tokens")
        input_ids = torch.randint(0, 1000, (1, 128))
        attention_mask = torch.ones(1, 128)
    else:
        tokens = tokenizer(text, return_tensors='pt')
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prosody_codes = prosody_codes.to(device)

    # Generate
    with torch.no_grad():
        audio = model.generate_with_facodec(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prosody_codes=prosody_codes,
        )

    return audio


def visualize_prosody_codes(
    prosody_codes: torch.Tensor,
    output_path: Optional[str] = None,
):
    """
    Visualize prosody codes as histogram and time series.

    Args:
        prosody_codes: [time] or [1, time] prosody codes
        output_path: Path to save visualization (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    codes = prosody_codes.squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Time series
    axes[0].plot(codes, 'b-', linewidth=0.5)
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Prosody Code')
    axes[0].set_title('FACodec Prosody Codes Over Time')
    axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(codes, bins=50, density=True, alpha=0.7, color='blue')
    axes[1].set_xlabel('Code Value')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Prosody Code Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech with FACodec prosody conditioning"
    )

    # Input options
    parser.add_argument(
        "--reference", "-r", type=str,
        help="Reference audio for prosody extraction"
    )
    parser.add_argument(
        "--prosody-file", type=str,
        help="Pre-extracted prosody codes (.pt file)"
    )
    parser.add_argument(
        "--speaker", type=str,
        help="Speaker reference audio (for prosody transfer)"
    )

    # Output options
    parser.add_argument(
        "--text", "-t", type=str,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output.wav",
        help="Output audio path"
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Only extract and save prosody codes"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize prosody codes"
    )

    # Model options
    parser.add_argument(
        "--checkpoint", type=str,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    # Validate args
    if not args.reference and not args.prosody_file:
        parser.error("Either --reference or --prosody-file required")

    if not args.extract_only and not args.text:
        parser.error("--text required for generation")

    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize FACodec extractor
    config = FACodecConfig()
    extractor = FACodecProsodyExtractor(config, use_official=True, device=device)
    extractor.to(device)
    extractor.eval()
    print("Loaded FACodec extractor")

    # Extract or load prosody
    if args.reference:
        print(f"Extracting prosody from: {args.reference}")
        features = extract_prosody_features(args.reference, extractor, device)
        prosody_codes = features['prosody_codes']
        prosody_emb = features['prosody_emb']
        speaker_emb = features['speaker_emb']

        print(f"  Prosody codes shape: {prosody_codes.shape}")
        print(f"  Prosody embedding shape: {prosody_emb.shape}")
        print(f"  Speaker embedding shape: {speaker_emb.shape}")

        if args.extract_only:
            # Save prosody codes
            output_path = args.output
            if not output_path.endswith('.pt'):
                output_path = output_path.rsplit('.', 1)[0] + '.pt'

            torch.save({
                'prosody_codes': prosody_codes.cpu(),
                'prosody_emb': prosody_emb.cpu(),
                'speaker_emb': speaker_emb.cpu(),
            }, output_path)
            print(f"Saved prosody features to: {output_path}")

            if args.visualize:
                viz_path = output_path.rsplit('.', 1)[0] + '_viz.png'
                visualize_prosody_codes(prosody_codes, viz_path)

            return

    elif args.prosody_file:
        print(f"Loading prosody from: {args.prosody_file}")
        loaded = torch.load(args.prosody_file, map_location=device)
        prosody_codes = loaded['prosody_codes']
        prosody_emb = loaded.get('prosody_emb')
        speaker_emb = loaded.get('speaker_emb')

        print(f"  Prosody codes shape: {prosody_codes.shape}")

    # Visualize if requested
    if args.visualize:
        visualize_prosody_codes(prosody_codes)

    # Extract speaker from separate reference if provided
    if args.speaker:
        print(f"Extracting speaker from: {args.speaker}")
        speaker_features = extract_prosody_features(args.speaker, extractor, device)
        speaker_emb = speaker_features['speaker_emb']
        print("  Using speaker embedding from separate reference")

    # Generate speech
    print(f"\nGenerating speech with text: \"{args.text}\"")

    # Load model if checkpoint provided
    if args.checkpoint:
        print(f"Loading model from: {args.checkpoint}")

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Create mock CSM for now (replace with real model)
        from training.train_facodec_prosody import MockCSM
        csm = MockCSM(hidden_size=config.hidden_size)
        model = FACodecControlledCSM(csm, config, freeze_csm=True)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Generate
        audio = generate_with_facodec_prosody(
            text=args.text,
            prosody_codes=prosody_codes,
            speaker_emb=speaker_emb,
            model=model,
            device=device,
        )

        # Save audio
        torchaudio.save(args.output, audio.cpu(), 16000)
        print(f"Saved audio to: {args.output}")

    else:
        print("\nNo checkpoint provided. Demo mode:")
        print("  - Prosody extraction complete")
        print("  - To generate speech, provide --checkpoint path")
        print("\nExample with checkpoint:")
        print(f"  python {sys.argv[0]} \\")
        print(f"    --reference {args.reference or 'reference.wav'} \\")
        print(f"    --text \"{args.text}\" \\")
        print(f"    --checkpoint ../checkpoints/facodec_prosody/best.pt \\")
        print(f"    --output {args.output}")

        # Show prosody statistics
        codes_np = prosody_codes.squeeze().cpu().numpy()
        print(f"\nProsody Statistics:")
        print(f"  Num frames: {len(codes_np)}")
        print(f"  Duration: ~{len(codes_np) * 0.0125:.2f}s")
        print(f"  Code range: [{codes_np.min()}, {codes_np.max()}]")
        print(f"  Code mean: {codes_np.mean():.2f}")
        print(f"  Code std: {codes_np.std():.2f}")

        # Save features for later use
        features_path = args.output.rsplit('.', 1)[0] + '_prosody.pt'
        torch.save({
            'prosody_codes': prosody_codes.cpu(),
            'prosody_emb': prosody_emb.cpu() if prosody_emb is not None else None,
            'speaker_emb': speaker_emb.cpu() if speaker_emb is not None else None,
            'text': args.text,
        }, features_path)
        print(f"\nSaved prosody features to: {features_path}")


if __name__ == "__main__":
    main()
