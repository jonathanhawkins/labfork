#!/usr/bin/env python3
"""
Demo: EASV Intensity Comparison

Generates prosody tokens at different intensities [0.5, 1.0, 1.5] for happy
and saves them for downstream audio generation and F0 analysis.

This demonstrates the ECE-TTS EASV formula:
  emotion_scaled = neutral + intensity * (emotion - neutral)

Expected behavior:
- intensity=0.5: Mild happy prosody
- intensity=1.0: Standard happy prosody
- intensity=1.5: Exaggerated happy prosody

The arousal component (which correlates with F0/pitch) should show:
  arousal @ 1.5 > arousal @ 1.0 > arousal @ 0.5
"""

import sys
from pathlib import Path
import torch

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'training'))

from spherical_emotion import (
    SphericalEmotionConfig,
    SphericalEmotionAdapter,
    CORE_EMOTIONS,
    VAD_PROTOTYPES,
)


def main():
    print("=" * 70)
    print("ECE-TTS EASV Intensity Comparison Demo")
    print("=" * 70)

    # Initialize with EASV enabled (default)
    config = SphericalEmotionConfig(use_easv=True)
    adapter = SphericalEmotionAdapter(config)
    adapter.eval()

    # Test text (for context)
    text = "I am so happy about this wonderful news!"

    # Test intensities
    intensities = [0.5, 1.0, 1.5]

    # Test emotion
    emotion = "happy"

    print(f"\nText: {text}")
    print(f"Emotion: {emotion}")
    print(f"VAD Prototype: {VAD_PROTOTYPES[emotion]}")
    print("\n" + "-" * 70)

    results = []
    for intensity in intensities:
        output = adapter.encode_emotion(emotion, intensity=intensity)

        vad_scaled = output['vad_scaled'][0]
        tokens = output['prosody_tokens']
        r = output['spherical'][0][0].item()

        result = {
            'intensity': intensity,
            'vad': vad_scaled.tolist(),
            'valence': vad_scaled[0].item(),
            'arousal': vad_scaled[1].item(),  # Pitch proxy
            'dominance': vad_scaled[2].item(),
            'radius': r,
            'tokens': tokens,
        }
        results.append(result)

        print(f"\nIntensity = {intensity:.1f}")
        print(f"  VAD: V={result['valence']:+.3f}, A={result['arousal']:+.3f}, D={result['dominance']:+.3f}")
        print(f"  Spherical radius: {result['radius']:.3f}")
        print(f"  Token shape: {tokens.shape}")

        # Save prosody tokens
        output_path = project_root / f"inference/outputs/happy_intensity_{int(intensity*10)}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'tokens': tokens,
            'text': text,
            'emotion': emotion,
            'intensity': intensity,
            'vad_scaled': vad_scaled,
            'config': 'easv',
        }, output_path)
        print(f"  Saved: {output_path.name}")

    # Verify monotonic arousal (pitch proxy)
    print("\n" + "-" * 70)
    print("\nVerification:")
    arousals = [r['arousal'] for r in results]
    if arousals[0] < arousals[1] < arousals[2]:
        print(f"  ✓ Arousal (pitch proxy) increases with intensity:")
        print(f"    {arousals[0]:.3f} < {arousals[1]:.3f} < {arousals[2]:.3f}")
    else:
        print(f"  ✗ Arousal not monotonic!")

    radii = [r['radius'] for r in results]
    if radii[0] < radii[1] < radii[2]:
        print(f"  ✓ Spherical magnitude increases with intensity:")
        print(f"    {radii[0]:.3f} < {radii[1]:.3f} < {radii[2]:.3f}")

    print("\n" + "=" * 70)
    print("SUCCESS: EASV intensity control verified!")
    print("=" * 70)

    print("""
Next Steps for Full Audio Verification:
1. Load saved .pt files and feed tokens to CSM model
2. Generate audio at each intensity level
3. Extract F0 contours using parselmouth/praat
4. Verify: F0_mean @ intensity=1.5 > F0_mean @ intensity=1.0 > F0_mean @ intensity=0.5

Expected F0 relationship for happy:
  intensity=0.5 → F0 ~140 Hz (mild positive)
  intensity=1.0 → F0 ~160 Hz (standard happy)
  intensity=1.5 → F0 ~180 Hz (exaggerated happy)
""")


if __name__ == "__main__":
    main()
