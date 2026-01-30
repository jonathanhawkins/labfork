"""
Inference with Hierarchical Emotion Distribution (HED) - V6

This script demonstrates how to use the HED-trained model for fine-grained
emotion control at word level during speech synthesis.

Example usage:
    python generate_with_hed.py \
        --checkpoint ../models/checkpoints/prosody_v6_hed/best.pt \
        --text "I am absolutely thrilled about this amazing news!" \
        --word-emotions "thrilled:happy:0.9,amazing:excited:0.8" \
        --output excited_speech.wav

The --word-emotions format is: "word:emotion:intensity,word:emotion:intensity"
Words not specified use neutral emotion at 0.5 intensity.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'training'))
sys.path.insert(0, str(project_root / 'backend'))

from prosody_conditioning import ProsodyConfig, ProsodyEncoder, TemporalProsodyEncoder
from hierarchical_emotion import HEDConfig, HierarchicalEmotionEncoder, HEDVarianceAdaptor


class HEDInference:
    """
    HED-based speech synthesis with word-level emotion control.

    This class loads a trained HED model and provides methods to:
    1. Specify per-word emotions for fine-grained control
    2. Generate prosody conditioning that respects word-level emotions
    3. Integrate with the prosody-conditioned CSM model
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cpu',
    ):
        self.device = torch.device(device)
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load configs from checkpoint
        prosody_cfg = self.checkpoint.get('prosody_config', {})
        self.prosody_config = ProsodyConfig(
            semantic_dim=prosody_cfg.get('semantic_dim', 8),
            acoustic_dim=prosody_cfg.get('acoustic_dim', 12),
            rhythm_dim=prosody_cfg.get('rhythm_dim', 8),
            contour_dim=prosody_cfg.get('contour_dim', 64),
            hidden_size=prosody_cfg.get('hidden_size', 2048),
            num_prosody_tokens=prosody_cfg.get('num_prosody_tokens', 4),
        )

        hed_cfg = self.checkpoint.get('hed_config', {})
        self.hed_config = HEDConfig(
            opensmile_dim=hed_cfg.get('opensmile_dim', 88),
            phoneme_hidden=hed_cfg.get('phoneme_hidden', 128),
            word_hidden=hed_cfg.get('word_hidden', 256),
            utterance_hidden=hed_cfg.get('utterance_hidden', 512),
            output_hidden=hed_cfg.get('output_hidden', 2048),
        )

        # Load models
        self._load_models()

        print(f"Loaded HED inference from: {checkpoint_path}")

    def _load_models(self):
        """Load prosody encoder and HED modules from checkpoint."""
        # Prosody encoder
        self.prosody_encoder = ProsodyEncoder(self.prosody_config).to(self.device)
        if 'prosody_encoder' in self.checkpoint:
            self.prosody_encoder.load_state_dict(self.checkpoint['prosody_encoder'])

        # Temporal encoder
        self.temporal_encoder = None
        if 'temporal_encoder' in self.checkpoint:
            self.temporal_encoder = TemporalProsodyEncoder(self.prosody_config).to(self.device)
            self.temporal_encoder.load_state_dict(self.checkpoint['temporal_encoder'])

        # HED encoder
        self.hed_encoder = None
        if 'hed_encoder' in self.checkpoint:
            self.hed_encoder = HierarchicalEmotionEncoder(self.hed_config).to(self.device)
            self.hed_encoder.load_state_dict(self.checkpoint['hed_encoder'])

        # HED variance adaptor
        self.hed_adaptor = None
        if 'hed_variance_adaptor' in self.checkpoint:
            self.hed_adaptor = HEDVarianceAdaptor(
                self.hed_config,
                prosody_hidden=self.prosody_config.hidden_size,
                num_tokens=self.prosody_config.num_prosody_tokens,
            ).to(self.device)
            self.hed_adaptor.load_state_dict(self.checkpoint['hed_variance_adaptor'])

        # Set to eval mode
        self.prosody_encoder.eval()
        if self.temporal_encoder:
            self.temporal_encoder.eval()
        if self.hed_encoder:
            self.hed_encoder.eval()
        if self.hed_adaptor:
            self.hed_adaptor.eval()

    # Emotion profiles for word-level control
    EMOTION_PROFILES = {
        'neutral': {'semantic_idx': 0, 'pitch': 0.5, 'energy': 0.5, 'rate': 0.5},
        'happy': {'semantic_idx': 1, 'pitch': 0.7, 'energy': 0.7, 'rate': 0.7},
        'sad': {'semantic_idx': 2, 'pitch': 0.3, 'energy': 0.3, 'rate': 0.3},
        'angry': {'semantic_idx': 3, 'pitch': 0.6, 'energy': 0.9, 'rate': 0.6},
        'surprised': {'semantic_idx': 4, 'pitch': 0.8, 'energy': 0.6, 'rate': 0.8},
        'fearful': {'semantic_idx': 4, 'pitch': 0.6, 'energy': 0.5, 'rate': 0.7},
        'calm': {'semantic_idx': 5, 'pitch': 0.4, 'energy': 0.4, 'rate': 0.4},
        'excited': {'semantic_idx': 6, 'pitch': 0.8, 'energy': 0.9, 'rate': 0.9},
    }

    def parse_word_emotions(self, word_emotions_str: str) -> Dict[str, Dict]:
        """
        Parse word emotions string into dict.

        Format: "word1:emotion1:intensity1,word2:emotion2:intensity2"
        Example: "thrilled:happy:0.9,amazing:excited:0.8"

        Returns:
            Dict mapping word (lowercase) -> {emotion, intensity}
        """
        if not word_emotions_str:
            return {}

        result = {}
        for part in word_emotions_str.split(','):
            parts = part.strip().split(':')
            if len(parts) >= 2:
                word = parts[0].lower()
                emotion = parts[1].lower()
                intensity = float(parts[2]) if len(parts) >= 3 else 0.7
                result[word] = {'emotion': emotion, 'intensity': intensity}

        return result

    def get_word_prosody(
        self,
        word: str,
        word_emotions: Dict[str, Dict],
        default_emotion: str = 'neutral',
        default_intensity: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Get prosody vectors for a single word.

        Args:
            word: The word text
            word_emotions: Dict mapping words to emotion/intensity
            default_emotion: Emotion for unspecified words
            default_intensity: Intensity for unspecified words

        Returns:
            Dict with prosody tensors for this word
        """
        word_lower = word.lower()

        if word_lower in word_emotions:
            emotion = word_emotions[word_lower]['emotion']
            intensity = word_emotions[word_lower]['intensity']
        else:
            emotion = default_emotion
            intensity = default_intensity

        profile = self.EMOTION_PROFILES.get(emotion, self.EMOTION_PROFILES['neutral'])

        # Build prosody vectors
        semantic = torch.zeros(self.prosody_config.semantic_dim)
        semantic[profile['semantic_idx']] = intensity

        acoustic = torch.zeros(self.prosody_config.acoustic_dim)
        acoustic[0] = profile['pitch'] * intensity  # pitch_mean
        acoustic[1] = 0.3 * intensity  # pitch_std
        acoustic[2] = profile['energy'] * intensity  # energy
        acoustic[3:] = 0.5

        rhythm = torch.zeros(self.prosody_config.rhythm_dim)
        rhythm[0] = profile['rate'] * intensity  # speaking_rate
        rhythm[1:] = 0.5

        contour = torch.ones(self.prosody_config.contour_dim) * profile['pitch'] * intensity

        return {
            'semantic': semantic,
            'acoustic': acoustic,
            'rhythm': rhythm,
            'contour': contour,
            'emotion': emotion,
            'intensity': intensity,
        }

    def generate_word_level_prosody(
        self,
        text: str,
        word_emotions: Dict[str, Dict],
        default_emotion: str = 'neutral',
    ) -> Dict[str, torch.Tensor]:
        """
        Generate per-word prosody conditioning.

        Args:
            text: Full text to synthesize
            word_emotions: Dict mapping specific words to emotion/intensity
            default_emotion: Default emotion for unspecified words

        Returns:
            Dict with:
                - 'word_prosodies': List of prosody dicts per word
                - 'temporal_prosody': Aggregated temporal tokens [num_words, hidden]
                - 'global_prosody': Averaged global tokens [1, hidden]
        """
        import re
        words = re.findall(r"[A-Za-z]+", text)

        if not words:
            words = ['']

        # Get prosody for each word
        word_prosodies = []
        for word in words:
            prosody = self.get_word_prosody(word, word_emotions, default_emotion)
            word_prosodies.append(prosody)

        # Stack word-level features
        semantic_stack = torch.stack([wp['semantic'] for wp in word_prosodies])
        acoustic_stack = torch.stack([wp['acoustic'] for wp in word_prosodies])
        rhythm_stack = torch.stack([wp['rhythm'] for wp in word_prosodies])
        contour_stack = torch.stack([wp['contour'] for wp in word_prosodies])

        # Add batch dimension
        semantic_batch = semantic_stack.unsqueeze(0).to(self.device)  # [1, num_words, dim]
        acoustic_batch = acoustic_stack.unsqueeze(0).to(self.device)
        rhythm_batch = rhythm_stack.unsqueeze(0).to(self.device)
        contour_batch = contour_stack.unsqueeze(0).to(self.device)

        # Generate temporal prosody tokens (one per word, up to num_prosody_tokens)
        num_words = len(words)
        num_segments = min(num_words, self.prosody_config.num_prosody_tokens)

        # Aggregate words into segments
        segment_size = max(1, num_words // num_segments)
        temporal_segments = {
            'semantic': [],
            'acoustic': [],
            'rhythm': [],
            'contour': [],
        }

        for seg_idx in range(num_segments):
            start_word = seg_idx * segment_size
            end_word = num_words if seg_idx == num_segments - 1 else start_word + segment_size

            temporal_segments['semantic'].append(semantic_stack[start_word:end_word].mean(dim=0))
            temporal_segments['acoustic'].append(acoustic_stack[start_word:end_word].mean(dim=0))
            temporal_segments['rhythm'].append(rhythm_stack[start_word:end_word].mean(dim=0))
            temporal_segments['contour'].append(contour_stack[start_word:end_word].mean(dim=0))

        # Stack segments
        temporal_prosody = {
            'semantic': torch.stack(temporal_segments['semantic']).unsqueeze(0).to(self.device),
            'acoustic': torch.stack(temporal_segments['acoustic']).unsqueeze(0).to(self.device),
            'rhythm': torch.stack(temporal_segments['rhythm']).unsqueeze(0).to(self.device),
            'contour': torch.stack(temporal_segments['contour']).unsqueeze(0).to(self.device),
        }

        # Generate global prosody (averaged)
        global_prosody = {
            'semantic': semantic_stack.mean(dim=0, keepdim=True).unsqueeze(0).to(self.device),
            'acoustic': acoustic_stack.mean(dim=0, keepdim=True).unsqueeze(0).to(self.device),
            'rhythm': rhythm_stack.mean(dim=0, keepdim=True).unsqueeze(0).to(self.device),
            'contour': contour_stack.mean(dim=0, keepdim=True).unsqueeze(0).to(self.device),
        }

        return {
            'word_prosodies': word_prosodies,
            'temporal_prosody': temporal_prosody,
            'global_prosody': global_prosody,
            'words': words,
        }

    @torch.no_grad()
    def get_prosody_tokens(
        self,
        text: str,
        word_emotions: Dict[str, Dict],
        use_temporal: bool = True,
    ) -> torch.Tensor:
        """
        Get prosody prefix tokens for CSM model.

        Args:
            text: Text to synthesize
            word_emotions: Word-level emotion specs
            use_temporal: Whether to use temporal encoder for per-segment control

        Returns:
            Prosody tokens [1, num_tokens, hidden]
        """
        prosody_data = self.generate_word_level_prosody(text, word_emotions)

        if use_temporal and self.temporal_encoder is not None:
            # Use temporal encoder for per-segment control
            temporal_prosody = prosody_data['temporal_prosody']
            tokens = self.temporal_encoder(
                temporal_prosody['semantic'],
                temporal_prosody['acoustic'],
                temporal_prosody['rhythm'],
                temporal_prosody['contour'],
            )
        else:
            # Use global prosody encoder
            global_prosody = prosody_data['global_prosody']
            tokens = self.prosody_encoder(
                global_prosody['semantic'].squeeze(1),
                global_prosody['acoustic'].squeeze(1),
                global_prosody['rhythm'].squeeze(1),
                global_prosody['contour'].squeeze(1),
            )

        return tokens

    def print_word_emotions(
        self,
        text: str,
        word_emotions: Dict[str, Dict],
    ):
        """Print visualization of word-level emotions."""
        import re
        words = re.findall(r"[A-Za-z]+", text)

        print("\nWord-Level Emotion Mapping:")
        print("-" * 50)
        for word in words:
            word_lower = word.lower()
            if word_lower in word_emotions:
                emotion = word_emotions[word_lower]['emotion']
                intensity = word_emotions[word_lower]['intensity']
                bar = '=' * int(intensity * 10)
                print(f"  {word:15} -> {emotion:10} [{bar:<10}] {intensity:.1f}")
            else:
                print(f"  {word:15} -> neutral    [=====     ] 0.5")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description='Generate speech with word-level emotion control (HED)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to HED checkpoint')
    parser.add_argument('--text', type=str, required=True, help='Text to synthesize')
    parser.add_argument('--word-emotions', type=str, default='',
                       help='Word emotions: "word1:emotion:intensity,word2:emotion:intensity"')
    parser.add_argument('--default-emotion', type=str, default='neutral',
                       help='Default emotion for unspecified words')
    parser.add_argument('--output', type=str, default='output_hed.wav', help='Output audio path')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda/mps)')
    parser.add_argument('--visualize', action='store_true', help='Print emotion visualization')
    args = parser.parse_args()

    # Load HED inference
    hed = HEDInference(args.checkpoint, device=args.device)

    # Parse word emotions
    word_emotions = hed.parse_word_emotions(args.word_emotions)

    # Print visualization
    if args.visualize or True:  # Always show for demo
        hed.print_word_emotions(args.text, word_emotions)

    # Get prosody tokens
    prosody_tokens = hed.get_prosody_tokens(args.text, word_emotions)
    print(f"\nProsody tokens shape: {prosody_tokens.shape}")

    # Generate word-level prosody data
    prosody_data = hed.generate_word_level_prosody(args.text, word_emotions, args.default_emotion)

    print(f"\nGenerated prosody for {len(prosody_data['words'])} words")
    print(f"Temporal segments: {prosody_data['temporal_prosody']['semantic'].shape[1]}")

    print("\n[NOTE] This script demonstrates prosody token generation.")
    print("       To generate actual audio, integrate with CSM model:")
    print("""
    from transformers import CsmForConditionalGeneration

    csm = CsmForConditionalGeneration.from_pretrained('sesame/csm-1b')

    # Get text embeddings
    text_embeds = csm.embed_text_tokens(tokenized_text)

    # Prepend prosody tokens
    combined = torch.cat([prosody_tokens, text_embeds], dim=1)

    # Generate audio
    audio = csm.generate(inputs_embeds=combined, ...)
    """)

    # Save prosody tokens for later use
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    prosody_path = Path(args.output).with_suffix('.prosody.pt')
    torch.save({
        'tokens': prosody_tokens,
        'text': args.text,
        'word_emotions': word_emotions,
        'words': prosody_data['words'],
        'temporal_prosody': prosody_data['temporal_prosody'],
        'global_prosody': prosody_data['global_prosody'],
    }, prosody_path)
    print(f"\nSaved prosody data to: {prosody_path}")


if __name__ == "__main__":
    main()
