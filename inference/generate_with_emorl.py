#!/usr/bin/env python3
"""
Generate speech with EMORL-TTS: VAD + Local Emphasis Control

Generates speech with both:
1. Global emotion intensity (via VAD coordinates)
2. Local word-level emphasis (specific words emphasized)

Features:
- Emphasize specific words in a sentence
- Control global emotion intensity independently
- Verify emphasis via energy/F0 contour analysis
- Generate samples for comparison

Usage:
    # Generate with emphasis on word 2
    python generate_with_emorl.py \
        --checkpoint ../checkpoints/emorl_tts/best.pt \
        --text "The quick brown fox jumps over the lazy dog" \
        --emphasis_words 2 4 \
        --emphasis_strengths 4 3 \
        --emotion happy \
        --intensity 0.8

    # Generate comparison samples (same sentence, different emphasis)
    python generate_with_emorl.py \
        --checkpoint ../checkpoints/emorl_tts/best.pt \
        --compare \
        --text "She sells seashells by the seashore"

    # Analyze emphasis in existing audio
    python generate_with_emorl.py --analyze output.wav
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

# =============================================================================
# VAD PROTOTYPES
# =============================================================================

VAD_PROTOTYPES = {
    'neutral': (0.0, 0.0, 0.0),
    'happy': (0.8, 0.6, 0.6),
    'joy': (0.9, 0.7, 0.5),
    'excited': (0.7, 0.9, 0.6),
    'sad': (-0.6, -0.4, -0.5),
    'angry': (-0.5, 0.8, 0.7),
    'fearful': (-0.7, 0.7, -0.7),
    'surprised': (0.3, 0.8, 0.2),
    'disgusted': (-0.6, 0.3, 0.4),
    'calm': (0.4, -0.5, 0.3),
}


def get_vad_for_emotion(emotion: str, intensity: float = 1.0) -> torch.Tensor:
    """Get VAD coordinates for an emotion, scaled by intensity."""
    vad = VAD_PROTOTYPES.get(emotion.lower(), VAD_PROTOTYPES['neutral'])
    vad_tensor = torch.tensor(vad, dtype=torch.float32)
    return vad_tensor * intensity


# =============================================================================
# EMPHASIS ANALYZER
# =============================================================================

class EmphasisAnalyzer:
    """Analyzes audio to detect emphasis patterns via energy and F0."""

    def __init__(self):
        self.sample_rate = 24000

    def extract_energy_contour(
        self,
        audio: np.ndarray,
        frame_length: int = 512,
        hop_length: int = 256,
    ) -> np.ndarray:
        """Extract energy contour from audio."""
        # Simple RMS energy
        num_frames = (len(audio) - frame_length) // hop_length + 1
        energy = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        return energy

    def extract_f0_contour(
        self,
        audio: np.ndarray,
        frame_length: int = 1024,
        hop_length: int = 256,
    ) -> np.ndarray:
        """Extract F0 contour using autocorrelation."""
        try:
            import librosa
            f0, _, _ = librosa.pyin(
                audio.astype(np.float32),
                fmin=50,
                fmax=500,
                sr=self.sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            # Replace NaN with 0
            f0 = np.nan_to_num(f0, nan=0.0)
            return f0
        except ImportError:
            # Fallback to simple autocorrelation
            num_frames = (len(audio) - frame_length) // hop_length + 1
            f0 = np.zeros(num_frames)

            for i in range(num_frames):
                start = i * hop_length
                frame = audio[start:start + frame_length]
                # Simple autocorrelation
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                # Find first peak after lag 0
                if len(autocorr) > 50:
                    peak_idx = np.argmax(autocorr[30:300]) + 30
                    if autocorr[peak_idx] > 0.3 * autocorr[0]:
                        f0[i] = self.sample_rate / peak_idx

            return f0

    def find_emphasis_peaks(
        self,
        energy: np.ndarray,
        f0: np.ndarray,
        num_words: int = 8,
    ) -> List[Dict]:
        """Find emphasis peaks from energy and F0 contours."""
        # Segment into word-like regions
        segment_len = len(energy) // num_words

        peaks = []
        for i in range(num_words):
            start = i * segment_len
            end = min((i + 1) * segment_len, len(energy))

            if start >= end:
                continue

            # Get segment statistics
            energy_seg = energy[start:end]
            f0_seg = f0[start:min(end, len(f0))] if len(f0) > 0 else np.zeros(1)

            energy_mean = np.mean(energy_seg)
            energy_max = np.max(energy_seg)
            f0_mean = np.mean(f0_seg[f0_seg > 0]) if np.any(f0_seg > 0) else 0
            f0_max = np.max(f0_seg) if len(f0_seg) > 0 else 0

            peaks.append({
                'word_idx': i,
                'energy_mean': float(energy_mean),
                'energy_max': float(energy_max),
                'f0_mean': float(f0_mean),
                'f0_max': float(f0_max),
            })

        return peaks

    def compute_emphasis_score(
        self,
        peaks: List[Dict],
    ) -> List[float]:
        """Compute emphasis score for each word position."""
        if not peaks:
            return []

        # Normalize energy and F0
        energy_vals = [p['energy_max'] for p in peaks]
        f0_vals = [p['f0_max'] for p in peaks]

        energy_mean = np.mean(energy_vals)
        energy_std = np.std(energy_vals) + 1e-8
        f0_mean = np.mean([f for f in f0_vals if f > 0]) if any(f > 0 for f in f0_vals) else 1
        f0_std = np.std([f for f in f0_vals if f > 0]) + 1e-8 if any(f > 0 for f in f0_vals) else 1

        scores = []
        for p in peaks:
            energy_z = (p['energy_max'] - energy_mean) / energy_std
            f0_z = (p['f0_max'] - f0_mean) / f0_std if p['f0_max'] > 0 else 0

            # Combined score (weighted)
            score = 0.6 * energy_z + 0.4 * f0_z
            scores.append(float(score))

        return scores

    def analyze_audio(
        self,
        audio_path: str,
        num_words: int = 8,
    ) -> Dict:
        """Full analysis of audio file for emphasis."""
        import scipy.io.wavfile as wav

        sample_rate, audio = wav.read(audio_path)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0

        if sample_rate != self.sample_rate:
            # Simple resampling
            import scipy.signal as signal
            num_samples = int(len(audio) * self.sample_rate / sample_rate)
            audio = signal.resample(audio, num_samples)

        # Extract contours
        energy = self.extract_energy_contour(audio)
        f0 = self.extract_f0_contour(audio)

        # Find peaks
        peaks = self.find_emphasis_peaks(energy, f0, num_words)

        # Compute scores
        scores = self.compute_emphasis_score(peaks)

        return {
            'energy_contour': energy.tolist(),
            'f0_contour': f0.tolist(),
            'word_peaks': peaks,
            'emphasis_scores': scores,
            'detected_emphasis_words': [
                i for i, s in enumerate(scores) if s > 1.0  # > 1 std above mean
            ],
        }


# =============================================================================
# EMORL GENERATOR
# =============================================================================

class EMORLGenerator:
    """Generate speech with EMORL-TTS (VAD + Local Emphasis)."""

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = self._setup_device(device)
        self.model = None
        self.config = None
        self.analyzer = EmphasisAnalyzer()

        self._load_model(checkpoint_path)

    def _setup_device(self, device: str = None) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _load_model(self, checkpoint_path: str):
        """Load EMORL-TTS model from checkpoint."""
        print(f"Loading model from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get config
        config_dict = checkpoint.get('config', {})

        # Import and create model
        from emorl_tts import EMORLConfig, EMORLPolicy
        from prosody_conditioning import ProsodyConfig, ProsodyEncoder

        self.config = EMORLConfig(**{
            k: v for k, v in config_dict.items()
            if hasattr(EMORLConfig, k)
        })

        # Create prosody encoder
        prosody_config = ProsodyConfig(hidden_size=self.config.hidden_size)
        prosody_encoder = ProsodyEncoder(prosody_config)

        # Create EMORL policy
        self.model = EMORLPolicy(prosody_encoder, self.config).to(self.device)

        # Load weights
        if 'policy' in checkpoint:
            self.model.load_state_dict(checkpoint['policy'])
            print("Loaded EMORL policy weights")
        else:
            print("Warning: No policy weights found in checkpoint")

        self.model.eval()
        print(f"Model loaded on {self.device}")

    def _text_to_prosody_features(
        self,
        text: str,
        emotion: str = 'neutral',
        intensity: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert text + emotion to prosody features.

        Note: In production, this would use a proper text frontend.
        Here we generate synthetic features for demonstration.
        """
        # Synthetic prosody features based on emotion
        vad = get_vad_for_emotion(emotion, intensity)

        # Semantic features (emotion scores)
        emotion_idx = list(VAD_PROTOTYPES.keys()).index(emotion) if emotion in VAD_PROTOTYPES else 0
        semantic = torch.zeros(8)
        semantic[emotion_idx] = intensity
        semantic[0] = 1.0 - intensity  # neutral baseline

        # Acoustic features (F0, energy based on VAD)
        arousal = abs(vad[1].item())
        acoustic = torch.tensor([
            0.5 + 0.3 * vad[1].item(),  # pitch (arousal affects)
            0.3 + 0.2 * arousal,  # pitch variance
            0.5 + 0.2 * intensity,  # energy
            0.4,  # HNR
            0.5 + 0.1 * vad[0].item(),  # spectral centroid
        ])

        # Rhythm features
        rhythm = torch.tensor([
            0.5 - 0.1 * arousal,  # speaking rate (slower if calm)
            0.2 + 0.1 * arousal,  # pause ratio
            0.3,  # syllable rate
            0.5,  # rhythm regularity
        ])

        # Contour (synthetic F0 trajectory)
        t = torch.linspace(0, 1, 32)
        base_f0 = 0.5 + 0.2 * torch.sin(2 * np.pi * t)
        contour = base_f0 + 0.1 * vad[1].item() * torch.randn(32)

        return {
            'semantic': semantic.unsqueeze(0).to(self.device),
            'acoustic': acoustic.unsqueeze(0).to(self.device),
            'rhythm': rhythm.unsqueeze(0).to(self.device),
            'contour': contour.unsqueeze(0).to(self.device),
            'target_vad': vad.unsqueeze(0).to(self.device),
            'target_intensity': torch.tensor([intensity]).to(self.device),
        }

    def generate(
        self,
        text: str,
        emotion: str = 'neutral',
        intensity: float = 0.7,
        emphasis_words: List[int] = None,
        emphasis_strengths: List[int] = None,
        output_path: str = None,
    ) -> Dict:
        """
        Generate speech with EMORL-TTS controls.

        Args:
            text: Text to synthesize
            emotion: Target emotion (happy, sad, angry, etc.)
            intensity: Global emotion intensity (0-1)
            emphasis_words: Word indices to emphasize (0-indexed)
            emphasis_strengths: Strength levels (0=none, 4=very strong)
            output_path: Path to save output WAV

        Returns:
            Dict with prosody_embedding, emphasis_analysis, etc.
        """
        # Get prosody features
        features = self._text_to_prosody_features(text, emotion, intensity)

        # Prepare emphasis targets
        max_emphasis = self.config.max_emphasis_words
        if emphasis_words is None:
            emphasis_words = []
        if emphasis_strengths is None:
            emphasis_strengths = [2] * len(emphasis_words)  # Default medium

        # Pad to max_emphasis_words
        positions = emphasis_words + [-1] * (max_emphasis - len(emphasis_words))
        strengths = emphasis_strengths + [0] * (max_emphasis - len(emphasis_strengths))

        positions = torch.tensor([positions[:max_emphasis]], device=self.device)
        strengths = torch.tensor([strengths[:max_emphasis]], device=self.device)

        # Generate prosody embedding
        with torch.no_grad():
            prosody_embedding, log_prob, value = self.model(
                features['semantic'],
                features['acoustic'],
                features['rhythm'],
                features['contour'],
                target_emphasis_positions=positions,
                target_emphasis_strengths=strengths,
                deterministic=True,
            )

        result = {
            'text': text,
            'emotion': emotion,
            'intensity': intensity,
            'emphasis_words': emphasis_words,
            'emphasis_strengths': emphasis_strengths[:len(emphasis_words)],
            'prosody_embedding': prosody_embedding.cpu().numpy(),
            'target_vad': features['target_vad'].cpu().numpy().tolist()[0],
        }

        # If output path provided, generate audio and analyze
        if output_path:
            # Note: Full audio generation requires TTS decoder
            # Here we save the prosody embedding for later use
            torch.save({
                'prosody_embedding': prosody_embedding.cpu(),
                'text': text,
                'emotion': emotion,
                'intensity': intensity,
                'emphasis_words': emphasis_words,
                'emphasis_strengths': emphasis_strengths[:len(emphasis_words)],
                'target_vad': features['target_vad'].cpu(),
            }, output_path.replace('.wav', '_prosody.pt'))

            print(f"Saved prosody to {output_path.replace('.wav', '_prosody.pt')}")
            result['prosody_path'] = output_path.replace('.wav', '_prosody.pt')

        return result

    def generate_comparison(
        self,
        text: str,
        emotion: str = 'neutral',
        intensity: float = 0.7,
        output_dir: str = 'outputs',
    ) -> List[Dict]:
        """
        Generate comparison samples with different emphasis patterns.

        Creates:
        - No emphasis (baseline)
        - Emphasis on word 1
        - Emphasis on word 3
        - Emphasis on word 5
        - Multi-word emphasis (words 1 and 5)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        words = text.split()
        num_words = len(words)

        comparison_patterns = [
            {'name': 'baseline', 'emphasis_words': [], 'emphasis_strengths': []},
            {'name': 'word_0', 'emphasis_words': [0], 'emphasis_strengths': [3]},
            {'name': 'word_2', 'emphasis_words': [min(2, num_words-1)], 'emphasis_strengths': [3]},
            {'name': 'word_4', 'emphasis_words': [min(4, num_words-1)], 'emphasis_strengths': [3]},
            {'name': 'multi', 'emphasis_words': [0, min(4, num_words-1)], 'emphasis_strengths': [3, 3]},
        ]

        results = []
        for pattern in comparison_patterns:
            output_path = f"{output_dir}/emorl_{emotion}_{pattern['name']}.wav"

            result = self.generate(
                text=text,
                emotion=emotion,
                intensity=intensity,
                emphasis_words=pattern['emphasis_words'],
                emphasis_strengths=pattern['emphasis_strengths'],
                output_path=output_path,
            )

            result['pattern_name'] = pattern['name']
            results.append(result)

            print(f"Generated {pattern['name']}: emphasis on words {pattern['emphasis_words']}")

        # Save comparison summary
        summary_path = f"{output_dir}/comparison_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'text': text,
                'emotion': emotion,
                'intensity': intensity,
                'patterns': [
                    {
                        'name': r['pattern_name'],
                        'emphasis_words': r['emphasis_words'],
                        'emphasis_strengths': r['emphasis_strengths'],
                        'target_vad': r['target_vad'],
                    }
                    for r in results
                ],
            }, f, indent=2)

        print(f"\nComparison summary saved to {summary_path}")
        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate speech with EMORL-TTS (VAD + Local Emphasis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with emphasis on specific words
  python generate_with_emorl.py \\
      --checkpoint ../checkpoints/emorl_tts/best.pt \\
      --text "The quick brown fox jumps over the lazy dog" \\
      --emphasis_words 2 4 \\
      --emphasis_strengths 4 3 \\
      --emotion happy

  # Generate comparison samples (different emphasis patterns)
  python generate_with_emorl.py \\
      --checkpoint ../checkpoints/emorl_tts/best.pt \\
      --text "She sells seashells by the seashore" \\
      --compare

  # Analyze existing audio for emphasis
  python generate_with_emorl.py --analyze output.wav

Emphasis strengths:
  0 = none
  1 = light
  2 = medium
  3 = strong
  4 = very strong
        """
    )

    parser.add_argument('--checkpoint', type=str, help='EMORL-TTS checkpoint path')
    parser.add_argument('--text', type=str, help='Text to synthesize')
    parser.add_argument('--emotion', type=str, default='neutral',
                       choices=list(VAD_PROTOTYPES.keys()),
                       help='Target emotion')
    parser.add_argument('--intensity', type=float, default=0.7,
                       help='Emotion intensity (0-1)')
    parser.add_argument('--emphasis_words', type=int, nargs='+',
                       help='Word indices to emphasize (0-indexed)')
    parser.add_argument('--emphasis_strengths', type=int, nargs='+',
                       help='Emphasis strengths (0-4) for each word')
    parser.add_argument('--output', type=str, default='output_emorl.wav',
                       help='Output path')
    parser.add_argument('--compare', action='store_true',
                       help='Generate comparison samples with different emphasis')
    parser.add_argument('--output_dir', type=str, default='outputs/emorl',
                       help='Output directory for comparison')
    parser.add_argument('--analyze', type=str,
                       help='Analyze existing audio file for emphasis')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    # Analysis mode
    if args.analyze:
        print(f"Analyzing {args.analyze} for emphasis patterns...")
        analyzer = EmphasisAnalyzer()
        analysis = analyzer.analyze_audio(args.analyze)

        print("\nEmphasis Analysis:")
        print("-" * 40)
        print(f"Detected emphasis on words: {analysis['detected_emphasis_words']}")
        print("\nPer-word emphasis scores:")
        for i, score in enumerate(analysis['emphasis_scores']):
            marker = " *" if score > 1.0 else ""
            print(f"  Word {i}: {score:+.2f}{marker}")

        # Save analysis
        analysis_path = args.analyze.replace('.wav', '_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nFull analysis saved to {analysis_path}")
        return

    # Generation mode
    if not args.checkpoint:
        print("Error: --checkpoint required for generation")
        return

    if not args.text:
        print("Error: --text required for generation")
        return

    # Create generator
    generator = EMORLGenerator(args.checkpoint, args.device)

    if args.compare:
        # Generate comparison samples
        print(f"\nGenerating comparison samples for: \"{args.text}\"")
        print(f"Emotion: {args.emotion}, Intensity: {args.intensity}")
        results = generator.generate_comparison(
            text=args.text,
            emotion=args.emotion,
            intensity=args.intensity,
            output_dir=args.output_dir,
        )
    else:
        # Generate single sample
        print(f"\nGenerating: \"{args.text}\"")
        print(f"Emotion: {args.emotion}, Intensity: {args.intensity}")
        print(f"Emphasis words: {args.emphasis_words}")
        print(f"Emphasis strengths: {args.emphasis_strengths}")

        result = generator.generate(
            text=args.text,
            emotion=args.emotion,
            intensity=args.intensity,
            emphasis_words=args.emphasis_words,
            emphasis_strengths=args.emphasis_strengths,
            output_path=args.output,
        )

        print(f"\nTarget VAD: {result['target_vad']}")
        print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
