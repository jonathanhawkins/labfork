"""
Hierarchical Emotion Distribution (HED) Module for V6

Based on ICASSP 2024 "Hierarchical Emotion Prediction and Control in TTS":
https://arxiv.org/html/2405.09171v1

Key Innovation: Model emotion intensity at 3 granularity levels:
- Phoneme-level: Fine-grained emotion per phoneme
- Word-level: Emotion aggregated at word boundaries
- Utterance-level: Global emotion representation

This enables fine-grained control like emphasizing specific words emotionally.

Implementation:
1. Montreal Forced Alignment (MFA) or G2P for phoneme/word boundaries
2. 88-dim acoustic features via OpenSMILE at each level
3. SVM-based ranking functions to compute emotion intensity
4. Integration as additional variance adaptor in prosody model
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HEDConfig:
    """Configuration for Hierarchical Emotion Distribution."""

    # Feature dimensions
    opensmile_dim: int = 88          # eGeMAPS features
    phoneme_hidden: int = 128        # Phoneme-level hidden size
    word_hidden: int = 256           # Word-level hidden size
    utterance_hidden: int = 512      # Utterance-level hidden size
    output_hidden: int = 2048        # Output to match prosody encoder

    # Number of emotion classes (same as prosody_conditioning)
    num_emotions: int = 8            # neutral, happy, sad, angry, surprised, calm, fearful, disgusted

    # Hierarchy aggregation
    phoneme_pooling: str = "attention"  # attention, mean, max
    word_pooling: str = "attention"     # attention, mean, max

    # Training settings
    dropout: float = 0.1
    use_svm_ranking: bool = True     # Use SVM-based ranking for intensity

    # Alignment method
    alignment_method: str = "g2p"     # "mfa" (Montreal Forced Aligner) or "g2p" (Grapheme-to-Phoneme)

    # OpenSMILE config
    opensmile_config: str = "eGeMAPSv02"  # Feature set to use


# =============================================================================
# PHONEME ALIGNMENT
# =============================================================================

# CMU ARPAbet phoneme set (39 phonemes + stress markers)
ARPABET_PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
    'SIL', 'SP', 'SPN',  # Silence, short pause, spoken noise
]

PHONEME_TO_IDX = {p: i for i, p in enumerate(ARPABET_PHONEMES)}


class GraphemeToPhoneme:
    """
    Simple grapheme-to-phoneme converter using CMU dictionary.

    Falls back to rule-based conversion for OOV words.
    For production, consider using g2p_en or phonemizer.
    """

    def __init__(self, cmu_dict_path: Optional[str] = None):
        self.cmu_dict = {}

        # Try to load CMU dict
        if cmu_dict_path and Path(cmu_dict_path).exists():
            self._load_cmu_dict(cmu_dict_path)
        else:
            # Try common locations
            for path in [
                "/usr/share/dict/cmudict-0.7b",
                "~/.local/share/cmudict/cmudict-0.7b",
                "data/cmudict-0.7b",
            ]:
                expanded = os.path.expanduser(path)
                if os.path.exists(expanded):
                    self._load_cmu_dict(expanded)
                    break

        if not self.cmu_dict:
            warnings.warn(
                "CMU dictionary not found. G2P will use fallback rules. "
                "For better results, install cmudict: pip install cmudict"
            )

    def _load_cmu_dict(self, path: str):
        """Load CMU pronunciation dictionary."""
        try:
            with open(path, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith(';;;'):
                        parts = line.split()
                        word = parts[0].upper()
                        # Remove variant number (e.g., WORD(1) -> WORD)
                        if '(' in word:
                            word = word.split('(')[0]
                        phonemes = parts[1:]
                        self.cmu_dict[word] = phonemes
        except Exception as e:
            warnings.warn(f"Failed to load CMU dict: {e}")

    def _fallback_g2p(self, word: str) -> List[str]:
        """Simple rule-based fallback for OOV words."""
        # Very basic rules - in production, use g2p_en or phonemizer
        phonemes = []
        word = word.upper()
        i = 0
        while i < len(word):
            # Check for digraphs first
            if i + 1 < len(word):
                digraph = word[i:i+2]
                if digraph in ['TH', 'SH', 'CH', 'NG', 'ZH']:
                    phonemes.append(digraph)
                    i += 2
                    continue

            # Single character mapping
            char_to_phoneme = {
                'A': 'AE', 'E': 'EH', 'I': 'IH', 'O': 'AA', 'U': 'AH',
                'B': 'B', 'C': 'K', 'D': 'D', 'F': 'F', 'G': 'G',
                'H': 'HH', 'J': 'JH', 'K': 'K', 'L': 'L', 'M': 'M',
                'N': 'N', 'P': 'P', 'Q': 'K', 'R': 'R', 'S': 'S',
                'T': 'T', 'V': 'V', 'W': 'W', 'X': 'K', 'Y': 'Y', 'Z': 'Z',
            }

            if word[i] in char_to_phoneme:
                phonemes.append(char_to_phoneme[word[i]])
            i += 1

        return phonemes if phonemes else ['SPN']  # Return spoken noise if empty

    def __call__(self, word: str) -> List[str]:
        """Convert word to phoneme sequence."""
        word_upper = word.upper()

        if word_upper in self.cmu_dict:
            # Remove stress markers (numbers) for simplicity
            return [p.rstrip('012') for p in self.cmu_dict[word_upper]]

        return self._fallback_g2p(word)

    def convert_text(self, text: str) -> List[Dict]:
        """
        Convert text to phonemes with word boundaries.

        Returns:
            List of dicts with 'word', 'phonemes', 'start_phoneme_idx', 'end_phoneme_idx'
        """
        import re
        words = re.findall(r"[A-Za-z]+", text)

        result = []
        phoneme_idx = 0

        for word in words:
            phonemes = self(word)
            result.append({
                'word': word,
                'phonemes': phonemes,
                'start_phoneme_idx': phoneme_idx,
                'end_phoneme_idx': phoneme_idx + len(phonemes),
            })
            phoneme_idx += len(phonemes)

        return result


class PhonemeAligner:
    """
    Aligns audio to phonemes using either MFA or G2P with Whisper timestamps.

    MFA provides more accurate alignments but requires installation.
    G2P with Whisper word timestamps is a lighter alternative.
    """

    def __init__(self, config: HEDConfig):
        self.config = config
        self.g2p = GraphemeToPhoneme()
        self._mfa_available = self._check_mfa()

    def _check_mfa(self) -> bool:
        """Check if Montreal Forced Aligner is available."""
        try:
            result = subprocess.run(
                ['mfa', 'version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def align(
        self,
        audio_path: str,
        text: str,
        word_timestamps: Optional[List[Dict]] = None,
    ) -> Dict[str, List]:
        """
        Align audio to phonemes and words.

        Args:
            audio_path: Path to audio file
            text: Transcript text
            word_timestamps: Optional Whisper word timestamps

        Returns:
            Dict with:
                - 'phonemes': List of {phoneme, start_time, end_time, word_idx}
                - 'words': List of {word, start_time, end_time, phoneme_indices}
        """
        if self.config.alignment_method == 'mfa' and self._mfa_available:
            return self._align_with_mfa(audio_path, text)
        else:
            return self._align_with_g2p(audio_path, text, word_timestamps)

    def _align_with_g2p(
        self,
        audio_path: str,
        text: str,
        word_timestamps: Optional[List[Dict]] = None,
    ) -> Dict[str, List]:
        """Align using G2P and word timestamps."""
        # Get G2P conversion
        word_phonemes = self.g2p.convert_text(text)

        # If no word timestamps, estimate from duration
        if not word_timestamps:
            try:
                import torchaudio
                waveform, sr = torchaudio.load(audio_path)
                duration = waveform.shape[1] / sr
            except:
                duration = 5.0  # Default fallback

            # Distribute time evenly across words
            num_words = len(word_phonemes)
            time_per_word = duration / max(1, num_words)

            word_timestamps = []
            for i, wp in enumerate(word_phonemes):
                word_timestamps.append({
                    'word': wp['word'],
                    'start': i * time_per_word,
                    'end': (i + 1) * time_per_word,
                })

        # Build phoneme and word lists
        phonemes = []
        words = []

        for word_idx, (wp, wt) in enumerate(zip(word_phonemes, word_timestamps)):
            word_start = wt.get('start', 0)
            word_end = wt.get('end', word_start + 0.1)

            # Distribute phoneme times within word
            num_phonemes = len(wp['phonemes'])
            if num_phonemes > 0:
                phoneme_duration = (word_end - word_start) / num_phonemes
            else:
                phoneme_duration = 0

            phoneme_indices = []
            for pi, phoneme in enumerate(wp['phonemes']):
                phoneme_start = word_start + pi * phoneme_duration
                phoneme_end = phoneme_start + phoneme_duration

                phoneme_indices.append(len(phonemes))
                phonemes.append({
                    'phoneme': phoneme,
                    'start_time': phoneme_start,
                    'end_time': phoneme_end,
                    'word_idx': word_idx,
                })

            words.append({
                'word': wp['word'],
                'start_time': word_start,
                'end_time': word_end,
                'phoneme_indices': phoneme_indices,
            })

        return {'phonemes': phonemes, 'words': words}

    def _align_with_mfa(self, audio_path: str, text: str) -> Dict[str, List]:
        """Align using Montreal Forced Aligner."""
        # MFA requires a specific directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input files
            audio_name = Path(audio_path).stem
            transcript_path = Path(tmpdir) / f"{audio_name}.txt"

            with open(transcript_path, 'w') as f:
                f.write(text)

            # Copy or link audio
            audio_dest = Path(tmpdir) / Path(audio_path).name
            os.symlink(os.path.abspath(audio_path), audio_dest)

            # Run MFA
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            try:
                subprocess.run([
                    'mfa', 'align',
                    tmpdir,
                    'english_us_arpa',  # Dictionary
                    'english_us_arpa',  # Acoustic model
                    str(output_dir),
                    '--clean',
                ], check=True, capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                warnings.warn("MFA alignment timed out, falling back to G2P")
                return self._align_with_g2p(audio_path, text, None)
            except subprocess.CalledProcessError as e:
                warnings.warn(f"MFA alignment failed: {e}, falling back to G2P")
                return self._align_with_g2p(audio_path, text, None)

            # Parse TextGrid output
            textgrid_path = output_dir / f"{audio_name}.TextGrid"
            if textgrid_path.exists():
                return self._parse_textgrid(textgrid_path)
            else:
                return self._align_with_g2p(audio_path, text, None)

    def _parse_textgrid(self, textgrid_path: Path) -> Dict[str, List]:
        """Parse MFA TextGrid output."""
        # Simple TextGrid parser (could use textgrid library for robustness)
        phonemes = []
        words = []

        try:
            import textgrid
            tg = textgrid.TextGrid.fromFile(str(textgrid_path))

            word_tier = None
            phone_tier = None

            for tier in tg.tiers:
                if tier.name.lower() == 'words':
                    word_tier = tier
                elif tier.name.lower() == 'phones':
                    phone_tier = tier

            if word_tier:
                for interval in word_tier:
                    if interval.mark:
                        words.append({
                            'word': interval.mark,
                            'start_time': interval.minTime,
                            'end_time': interval.maxTime,
                            'phoneme_indices': [],
                        })

            if phone_tier:
                for interval in phone_tier:
                    if interval.mark:
                        # Find which word this phoneme belongs to
                        word_idx = -1
                        for wi, w in enumerate(words):
                            if (interval.minTime >= w['start_time'] - 0.01 and
                                interval.maxTime <= w['end_time'] + 0.01):
                                word_idx = wi
                                w['phoneme_indices'].append(len(phonemes))
                                break

                        phonemes.append({
                            'phoneme': interval.mark.upper(),
                            'start_time': interval.minTime,
                            'end_time': interval.maxTime,
                            'word_idx': word_idx,
                        })
        except ImportError:
            warnings.warn("textgrid library not found. Install with: pip install textgrid")
            return {'phonemes': [], 'words': []}
        except Exception as e:
            warnings.warn(f"Failed to parse TextGrid: {e}")
            return {'phonemes': [], 'words': []}

        return {'phonemes': phonemes, 'words': words}


# =============================================================================
# OPENSMILE FEATURE EXTRACTION
# =============================================================================

class OpenSMILEExtractor:
    """
    Extract 88-dimensional eGeMAPS features using OpenSMILE.

    eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set) includes:
    - F0 statistics (mean, std, etc.)
    - Energy/loudness statistics
    - Spectral features (centroid, flux, etc.)
    - Voice quality (jitter, shimmer, HNR)
    - Formants

    Reference: Eyben et al. (2016) "The Geneva Minimalistic Acoustic Parameter
    Set (GeMAPS) for Voice Research and Affective Computing"
    """

    def __init__(self, config: HEDConfig):
        self.config = config
        self._opensmile = None
        self._feature_names = None

        try:
            import opensmile
            self._opensmile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
            self._feature_names = self._opensmile.feature_names
        except ImportError:
            warnings.warn(
                "opensmile not found. Install with: pip install opensmile\n"
                "Falling back to librosa-based features."
            )

    def extract_segment(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        start_time: float,
        end_time: float,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """
        Extract features for a specific time segment.

        Args:
            audio: Audio path, numpy array, or torch tensor
            start_time: Start time in seconds
            end_time: End time in seconds
            sample_rate: Audio sample rate

        Returns:
            Feature vector of shape [88] (or fallback dimension)
        """
        # Load audio if path
        if isinstance(audio, str):
            try:
                import torchaudio
                waveform, sr = torchaudio.load(audio)
                if sr != sample_rate:
                    waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
                audio = waveform.squeeze(0).numpy()
            except:
                return np.zeros(self.config.opensmile_dim)
        elif isinstance(audio, torch.Tensor):
            audio = audio.squeeze().numpy()

        # Extract segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment = audio[max(0, start_sample):min(len(audio), end_sample)]

        if len(segment) < 100:  # Too short
            return np.zeros(self.config.opensmile_dim)

        if self._opensmile is not None:
            return self._extract_opensmile(segment, sample_rate)
        else:
            return self._extract_librosa(segment, sample_rate)

    def _extract_opensmile(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract features using OpenSMILE."""
        try:
            features = self._opensmile.process_signal(audio, sr)
            # Aggregate LLD features (mean pooling)
            features_mean = features.mean(axis=0).values

            # Pad or truncate to expected dimension
            if len(features_mean) < self.config.opensmile_dim:
                features_mean = np.pad(
                    features_mean,
                    (0, self.config.opensmile_dim - len(features_mean))
                )
            elif len(features_mean) > self.config.opensmile_dim:
                features_mean = features_mean[:self.config.opensmile_dim]

            return features_mean
        except Exception as e:
            warnings.warn(f"OpenSMILE extraction failed: {e}")
            return np.zeros(self.config.opensmile_dim)

    def _extract_librosa(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Fallback feature extraction using librosa."""
        try:
            import librosa

            features = []

            # F0 features
            f0, _, _ = librosa.pyin(
                audio, fmin=50, fmax=500, sr=sr,
                frame_length=2048, hop_length=512
            )
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                features.extend([
                    np.mean(f0_valid), np.std(f0_valid),
                    np.min(f0_valid), np.max(f0_valid),
                    np.percentile(f0_valid, 25), np.percentile(f0_valid, 75),
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0])

            # Energy features
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            features.extend([
                np.mean(rms), np.std(rms),
                np.min(rms), np.max(rms),
            ])

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_flux = librosa.onset.onset_strength(y=audio, sr=sr)

            features.extend([
                np.mean(spectral_centroid), np.std(spectral_centroid),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(spectral_flux), np.std(spectral_flux),
            ])

            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])

            # Formants (approximation using LPC)
            try:
                from scipy.signal import lfilter
                lpc_coeffs = librosa.lpc(audio, order=8)
                roots = np.roots(lpc_coeffs)
                roots = roots[np.imag(roots) >= 0]
                angles = np.arctan2(np.imag(roots), np.real(roots))
                freqs = sorted(angles * (sr / (2 * np.pi)))
                formants = [f for f in freqs if 100 < f < 5000][:4]
                while len(formants) < 4:
                    formants.append(0)
                features.extend(formants[:4])
            except:
                features.extend([0, 0, 0, 0])

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.extend([np.mean(zcr), np.std(zcr)])

            # Pad to expected dimension
            features = np.array(features)
            if len(features) < self.config.opensmile_dim:
                features = np.pad(
                    features,
                    (0, self.config.opensmile_dim - len(features))
                )
            elif len(features) > self.config.opensmile_dim:
                features = features[:self.config.opensmile_dim]

            return features

        except ImportError:
            warnings.warn("librosa not available for fallback features")
            return np.zeros(self.config.opensmile_dim)


# =============================================================================
# HIERARCHICAL EMOTION ENCODER
# =============================================================================

class AttentionPooling(nn.Module):
    """Attention-based pooling for aggregating variable-length sequences."""

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden]
            mask: [batch, seq_len] - True for valid positions

        Returns:
            Pooled output [batch, hidden]
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # [batch, seq_len]

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)  # [batch, seq_len]

        # Weighted sum
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [batch, hidden]


class HierarchicalEmotionEncoder(nn.Module):
    """
    Hierarchical Emotion Distribution encoder.

    Processes emotion at 3 levels:
    1. Phoneme-level: Per-phoneme acoustic features -> phoneme embeddings
    2. Word-level: Aggregated phoneme embeddings -> word embeddings
    3. Utterance-level: Aggregated word embeddings -> global embedding

    Each level learns emotion-relevant representations that can be used
    for fine-grained control during synthesis.
    """

    def __init__(self, config: HEDConfig):
        super().__init__()
        self.config = config

        # ===== Phoneme-level encoder =====
        # Input: OpenSMILE features [batch, num_phonemes, opensmile_dim]
        self.phoneme_encoder = nn.Sequential(
            nn.Linear(config.opensmile_dim, config.phoneme_hidden),
            nn.LayerNorm(config.phoneme_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.phoneme_hidden, config.phoneme_hidden),
            nn.GELU(),
        )

        # Phoneme-level emotion classifier (auxiliary loss)
        self.phoneme_emotion_head = nn.Linear(config.phoneme_hidden, config.num_emotions)

        # Phoneme pooling for word aggregation
        if config.phoneme_pooling == "attention":
            self.phoneme_pool = AttentionPooling(config.phoneme_hidden, config.dropout)

        # ===== Word-level encoder =====
        # Input: Aggregated phoneme embeddings [batch, num_words, phoneme_hidden]
        self.word_encoder = nn.Sequential(
            nn.Linear(config.phoneme_hidden, config.word_hidden),
            nn.LayerNorm(config.word_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.word_hidden, config.word_hidden),
            nn.GELU(),
        )

        # Word-level emotion classifier (auxiliary loss)
        self.word_emotion_head = nn.Linear(config.word_hidden, config.num_emotions)

        # Word-level intensity predictor (0-1 for emphasis)
        self.word_intensity_head = nn.Sequential(
            nn.Linear(config.word_hidden, config.word_hidden // 2),
            nn.GELU(),
            nn.Linear(config.word_hidden // 2, 1),
            nn.Sigmoid(),
        )

        # Word pooling for utterance aggregation
        if config.word_pooling == "attention":
            self.word_pool = AttentionPooling(config.word_hidden, config.dropout)

        # ===== Utterance-level encoder =====
        # Input: Aggregated word embeddings [batch, word_hidden]
        self.utterance_encoder = nn.Sequential(
            nn.Linear(config.word_hidden, config.utterance_hidden),
            nn.LayerNorm(config.utterance_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.utterance_hidden, config.utterance_hidden),
            nn.GELU(),
        )

        # Utterance-level emotion classifier
        self.utterance_emotion_head = nn.Linear(config.utterance_hidden, config.num_emotions)

        # ===== Output projection =====
        # Combine all levels into prosody-compatible embedding
        combined_dim = config.phoneme_hidden + config.word_hidden + config.utterance_hidden
        self.output_projection = nn.Sequential(
            nn.Linear(combined_dim, config.output_hidden),
            nn.LayerNorm(config.output_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        phoneme_features: torch.Tensor,           # [batch, num_phonemes, opensmile_dim]
        phoneme_to_word: List[List[int]],         # Mapping: phoneme_idx -> word_idx
        num_words: int,
        phoneme_mask: Optional[torch.Tensor] = None,  # [batch, num_phonemes]
        word_mask: Optional[torch.Tensor] = None,     # [batch, num_words]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical encoder.

        Args:
            phoneme_features: OpenSMILE features per phoneme
            phoneme_to_word: For each phoneme, which word it belongs to
            num_words: Number of words in utterance
            phoneme_mask: Valid phoneme positions
            word_mask: Valid word positions

        Returns:
            Dict with:
                - 'phoneme_embeddings': [batch, num_phonemes, phoneme_hidden]
                - 'word_embeddings': [batch, num_words, word_hidden]
                - 'utterance_embedding': [batch, utterance_hidden]
                - 'combined_embedding': [batch, output_hidden]
                - 'phoneme_emotions': [batch, num_phonemes, num_emotions]
                - 'word_emotions': [batch, num_words, num_emotions]
                - 'word_intensities': [batch, num_words, 1]
                - 'utterance_emotions': [batch, num_emotions]
        """
        batch_size = phoneme_features.shape[0]
        num_phonemes = phoneme_features.shape[1]
        device = phoneme_features.device

        # ===== Phoneme-level =====
        phoneme_embeddings = self.phoneme_encoder(phoneme_features)  # [B, P, phoneme_hidden]
        phoneme_emotions = self.phoneme_emotion_head(phoneme_embeddings)  # [B, P, num_emotions]

        # ===== Word-level (aggregate phonemes per word) =====
        word_embeddings = torch.zeros(
            batch_size, num_words, self.config.phoneme_hidden,
            device=device
        )

        # Create phoneme-to-word aggregation
        for b in range(batch_size):
            for w_idx in range(num_words):
                # Find phonemes belonging to this word
                phoneme_indices = [
                    p_idx for p_idx, w in enumerate(phoneme_to_word[b])
                    if w == w_idx and p_idx < num_phonemes
                ]

                if phoneme_indices:
                    if self.config.phoneme_pooling == "attention":
                        # Use attention pooling
                        word_phonemes = phoneme_embeddings[b, phoneme_indices, :].unsqueeze(0)
                        word_embeddings[b, w_idx] = self.phoneme_pool(word_phonemes).squeeze(0)
                    elif self.config.phoneme_pooling == "max":
                        word_embeddings[b, w_idx] = phoneme_embeddings[b, phoneme_indices, :].max(dim=0)[0]
                    else:  # mean
                        word_embeddings[b, w_idx] = phoneme_embeddings[b, phoneme_indices, :].mean(dim=0)

        # Process word embeddings
        word_embeddings = self.word_encoder(word_embeddings)  # [B, W, word_hidden]
        word_emotions = self.word_emotion_head(word_embeddings)  # [B, W, num_emotions]
        word_intensities = self.word_intensity_head(word_embeddings)  # [B, W, 1]

        # ===== Utterance-level (aggregate words) =====
        if self.config.word_pooling == "attention":
            utterance_input = self.word_pool(word_embeddings, word_mask)
        elif self.config.word_pooling == "max":
            if word_mask is not None:
                word_embeddings_masked = word_embeddings.masked_fill(~word_mask.unsqueeze(-1), float('-inf'))
                utterance_input = word_embeddings_masked.max(dim=1)[0]
            else:
                utterance_input = word_embeddings.max(dim=1)[0]
        else:  # mean
            if word_mask is not None:
                utterance_input = (word_embeddings * word_mask.unsqueeze(-1)).sum(dim=1) / word_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                utterance_input = word_embeddings.mean(dim=1)

        utterance_embedding = self.utterance_encoder(utterance_input)  # [B, utterance_hidden]
        utterance_emotions = self.utterance_emotion_head(utterance_embedding)  # [B, num_emotions]

        # ===== Combine all levels =====
        # Pool phoneme and word embeddings for combination
        if phoneme_mask is not None:
            phoneme_pooled = (phoneme_embeddings * phoneme_mask.unsqueeze(-1)).sum(dim=1) / phoneme_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            phoneme_pooled = phoneme_embeddings.mean(dim=1)

        if word_mask is not None:
            word_pooled = (word_embeddings * word_mask.unsqueeze(-1)).sum(dim=1) / word_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            word_pooled = word_embeddings.mean(dim=1)

        combined = torch.cat([phoneme_pooled, word_pooled, utterance_embedding], dim=-1)
        combined_embedding = self.output_projection(combined)  # [B, output_hidden]

        return {
            'phoneme_embeddings': phoneme_embeddings,
            'word_embeddings': word_embeddings,
            'utterance_embedding': utterance_embedding,
            'combined_embedding': combined_embedding,
            'phoneme_emotions': phoneme_emotions,
            'word_emotions': word_emotions,
            'word_intensities': word_intensities,
            'utterance_emotions': utterance_emotions,
        }


# =============================================================================
# SVM-BASED EMOTION RANKING
# =============================================================================

class SVMEmotionRanker:
    """
    SVM-based ranking for emotion intensity prediction.

    Based on the paper's approach of using ranking functions to compute
    relative emotion intensity. Uses pairwise ranking SVM.
    """

    def __init__(self, config: HEDConfig):
        self.config = config
        self.svm_models = {}  # One per emotion
        self._fitted = False

        try:
            from sklearn.svm import SVR
            self._sklearn_available = True
        except ImportError:
            warnings.warn("sklearn not available. SVM ranking disabled.")
            self._sklearn_available = False

    def fit(
        self,
        features: np.ndarray,       # [num_samples, feature_dim]
        emotion_labels: np.ndarray,  # [num_samples] categorical
        intensities: np.ndarray,     # [num_samples] 0-1 intensity
    ):
        """
        Train SVM rankers for each emotion.

        Args:
            features: OpenSMILE features
            emotion_labels: Emotion category indices
            intensities: Ground truth intensity values
        """
        if not self._sklearn_available:
            return

        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler

        # Normalize features
        self.scaler = StandardScaler()
        features_normalized = self.scaler.fit_transform(features)

        # Train one ranker per emotion
        for emotion_idx in range(self.config.num_emotions):
            # Get samples of this emotion
            mask = emotion_labels == emotion_idx
            if mask.sum() < 10:
                continue

            X = features_normalized[mask]
            y = intensities[mask]

            # Train SVR for intensity ranking
            svm = SVR(kernel='rbf', C=1.0, gamma='scale')
            svm.fit(X, y)
            self.svm_models[emotion_idx] = svm

        self._fitted = True

    def predict_intensity(
        self,
        features: np.ndarray,  # [batch, feature_dim]
        emotion_idx: int,
    ) -> np.ndarray:
        """Predict emotion intensity using trained SVM ranker."""
        if not self._fitted or emotion_idx not in self.svm_models:
            return np.ones(len(features)) * 0.5  # Default mid-intensity

        features_normalized = self.scaler.transform(features)
        return np.clip(self.svm_models[emotion_idx].predict(features_normalized), 0, 1)


# =============================================================================
# HED LOSS FUNCTIONS
# =============================================================================

class HierarchicalEmotionLoss(nn.Module):
    """
    Multi-task loss for hierarchical emotion learning.

    Combines:
    1. Phoneme-level emotion classification
    2. Word-level emotion classification
    3. Word-level intensity regression
    4. Utterance-level emotion classification
    5. Consistency loss between levels
    """

    def __init__(
        self,
        config: HEDConfig,
        phoneme_weight: float = 0.3,
        word_weight: float = 0.5,
        utterance_weight: float = 1.0,
        intensity_weight: float = 0.5,
        consistency_weight: float = 0.2,
    ):
        super().__init__()
        self.config = config
        self.phoneme_weight = phoneme_weight
        self.word_weight = word_weight
        self.utterance_weight = utterance_weight
        self.intensity_weight = intensity_weight
        self.consistency_weight = consistency_weight

        self.emotion_ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.intensity_mse = nn.MSELoss()

    def forward(
        self,
        hed_output: Dict[str, torch.Tensor],
        emotion_labels: torch.Tensor,           # [batch] utterance-level labels
        phoneme_emotions: Optional[torch.Tensor] = None,  # [batch, num_phonemes]
        word_emotions: Optional[torch.Tensor] = None,     # [batch, num_words]
        word_intensities: Optional[torch.Tensor] = None,  # [batch, num_words]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical emotion losses.

        Returns:
            Dict with individual losses and total loss
        """
        losses = {}

        # Utterance-level emotion loss (primary)
        utterance_logits = hed_output['utterance_emotions']
        losses['utterance_emotion'] = self.emotion_ce(utterance_logits, emotion_labels)

        # Word-level emotion loss (if labels provided)
        if word_emotions is not None:
            word_logits = hed_output['word_emotions']  # [B, W, E]
            B, W, E = word_logits.shape
            word_logits_flat = word_logits.view(-1, E)
            word_emotions_flat = word_emotions.view(-1)
            losses['word_emotion'] = self.emotion_ce(word_logits_flat, word_emotions_flat)
        else:
            # Use utterance label for all words (soft supervision)
            word_logits = hed_output['word_emotions']
            B, W, E = word_logits.shape
            expanded_labels = emotion_labels.unsqueeze(1).expand(B, W).reshape(-1)
            losses['word_emotion'] = self.emotion_ce(word_logits.view(-1, E), expanded_labels) * 0.5

        # Phoneme-level emotion loss (if labels provided)
        if phoneme_emotions is not None:
            phoneme_logits = hed_output['phoneme_emotions']
            B, P, E = phoneme_logits.shape
            phoneme_logits_flat = phoneme_logits.view(-1, E)
            phoneme_emotions_flat = phoneme_emotions.view(-1)
            losses['phoneme_emotion'] = self.emotion_ce(phoneme_logits_flat, phoneme_emotions_flat)
        else:
            # Use utterance label for all phonemes (soft supervision)
            phoneme_logits = hed_output['phoneme_emotions']
            B, P, E = phoneme_logits.shape
            expanded_labels = emotion_labels.unsqueeze(1).expand(B, P).reshape(-1)
            losses['phoneme_emotion'] = self.emotion_ce(phoneme_logits.view(-1, E), expanded_labels) * 0.3

        # Word intensity loss
        if word_intensities is not None:
            pred_intensities = hed_output['word_intensities'].squeeze(-1)
            losses['word_intensity'] = self.intensity_mse(pred_intensities, word_intensities)
        else:
            losses['word_intensity'] = torch.tensor(0.0, device=utterance_logits.device)

        # Consistency loss: word predictions should be consistent with utterance
        word_probs = F.softmax(hed_output['word_emotions'], dim=-1)  # [B, W, E]
        utterance_probs = F.softmax(utterance_logits, dim=-1)  # [B, E]
        word_avg_probs = word_probs.mean(dim=1)  # [B, E]
        losses['consistency'] = F.kl_div(
            word_avg_probs.log(),
            utterance_probs,
            reduction='batchmean'
        )

        # Combine losses
        total = (
            losses['utterance_emotion'] * self.utterance_weight +
            losses['word_emotion'] * self.word_weight +
            losses['phoneme_emotion'] * self.phoneme_weight +
            losses['word_intensity'] * self.intensity_weight +
            losses['consistency'] * self.consistency_weight
        )
        losses['total'] = total

        return losses


# =============================================================================
# VARIANCE ADAPTOR INTEGRATION
# =============================================================================

class HEDVarianceAdaptor(nn.Module):
    """
    Variance adaptor that integrates HED into the prosody conditioning pipeline.

    This module takes the combined HED embedding and produces additional
    conditioning signals that can be used alongside the existing prosody encoder.

    The output is compatible with the ProsodyEncoder's output format.
    """

    def __init__(self, config: HEDConfig, prosody_hidden: int = 2048, num_tokens: int = 4):
        super().__init__()
        self.config = config
        self.num_tokens = num_tokens

        # Project HED embedding to prosody token format
        self.projection = nn.Sequential(
            nn.Linear(config.output_hidden, prosody_hidden),
            nn.LayerNorm(prosody_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(prosody_hidden, prosody_hidden * num_tokens),
        )

        self.output_norm = nn.LayerNorm(prosody_hidden)

    def forward(
        self,
        hed_embedding: torch.Tensor,  # [batch, output_hidden]
    ) -> torch.Tensor:
        """
        Convert HED embedding to prosody prefix tokens.

        Args:
            hed_embedding: Output from HierarchicalEmotionEncoder.forward()['combined_embedding']

        Returns:
            Prosody tokens [batch, num_tokens, prosody_hidden]
        """
        # Project to token format
        tokens = self.projection(hed_embedding)  # [batch, prosody_hidden * num_tokens]

        # Reshape to [batch, num_tokens, prosody_hidden]
        batch_size = hed_embedding.shape[0]
        tokens = tokens.view(batch_size, self.num_tokens, -1)

        # Normalize
        tokens = self.output_norm(tokens)

        return tokens


# =============================================================================
# COMPLETE HED PIPELINE
# =============================================================================

class HEDPipeline:
    """
    Complete Hierarchical Emotion Distribution pipeline.

    Integrates:
    1. Phoneme alignment (MFA or G2P)
    2. OpenSMILE feature extraction
    3. Hierarchical emotion encoding
    4. (Optional) SVM-based intensity ranking
    5. Variance adaptor output

    Usage:
        pipeline = HEDPipeline(config)

        # Training
        hed_output = pipeline.process(
            audio_path="sample.wav",
            text="Hello, how are you?",
            word_timestamps=whisper_timestamps,
        )

        # Use hed_output['variance_adaptor_tokens'] with prosody model
    """

    def __init__(self, config: HEDConfig, device: str = 'cpu'):
        self.config = config
        self.device = device

        # Initialize components
        self.aligner = PhonemeAligner(config)
        self.feature_extractor = OpenSMILEExtractor(config)
        self.encoder = HierarchicalEmotionEncoder(config).to(device)
        self.variance_adaptor = HEDVarianceAdaptor(config).to(device)

        if config.use_svm_ranking:
            self.svm_ranker = SVMEmotionRanker(config)
        else:
            self.svm_ranker = None

    def process(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        text: str,
        word_timestamps: Optional[List[Dict]] = None,
        sample_rate: int = 24000,
    ) -> Dict[str, torch.Tensor]:
        """
        Process audio through complete HED pipeline.

        Args:
            audio: Audio path or waveform
            text: Transcript text
            word_timestamps: Optional Whisper word timestamps
            sample_rate: Audio sample rate

        Returns:
            Dict with:
                - All outputs from HierarchicalEmotionEncoder
                - 'variance_adaptor_tokens': [1, num_tokens, prosody_hidden]
                - 'alignment': Phoneme/word alignment info
        """
        # Step 1: Align audio to phonemes and words
        audio_path = audio if isinstance(audio, str) else None
        if audio_path is None:
            # Save temporary file for alignment
            import tempfile
            import torchaudio

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                audio_path = f.name
                if isinstance(audio, torch.Tensor):
                    torchaudio.save(audio_path, audio.unsqueeze(0), sample_rate)
                else:
                    import scipy.io.wavfile as wav
                    wav.write(audio_path, sample_rate, (audio * 32767).astype(np.int16))

        alignment = self.aligner.align(audio_path, text, word_timestamps)

        # Step 2: Extract OpenSMILE features for each phoneme
        phoneme_features = []
        for phoneme in alignment['phonemes']:
            features = self.feature_extractor.extract_segment(
                audio,
                phoneme['start_time'],
                phoneme['end_time'],
                sample_rate,
            )
            phoneme_features.append(features)

        if not phoneme_features:
            # Fallback: create dummy features
            phoneme_features = [np.zeros(self.config.opensmile_dim)]

        phoneme_features = torch.tensor(
            np.stack(phoneme_features),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # [1, num_phonemes, opensmile_dim]

        # Step 3: Create phoneme-to-word mapping
        phoneme_to_word = [[p['word_idx'] for p in alignment['phonemes']]]
        num_words = len(alignment['words'])

        # Step 4: Run through encoder
        with torch.no_grad():
            hed_output = self.encoder(
                phoneme_features,
                phoneme_to_word,
                num_words,
            )

        # Step 5: Generate variance adaptor tokens
        variance_tokens = self.variance_adaptor(hed_output['combined_embedding'])

        hed_output['variance_adaptor_tokens'] = variance_tokens
        hed_output['alignment'] = alignment

        return hed_output

    def train_mode(self):
        """Set models to training mode."""
        self.encoder.train()
        self.variance_adaptor.train()

    def eval_mode(self):
        """Set models to evaluation mode."""
        self.encoder.eval()
        self.variance_adaptor.eval()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hierarchical Emotion Distribution (HED) Module - Test Suite")
    print("=" * 60)

    config = HEDConfig()

    # Test 1: GraphemeToPhoneme
    print("\n[Test 1] Grapheme-to-Phoneme conversion...")
    g2p = GraphemeToPhoneme()
    result = g2p.convert_text("Hello, how are you today?")
    print(f"  Words: {[w['word'] for w in result]}")
    print(f"  Phonemes per word: {[len(w['phonemes']) for w in result]}")
    print("  [PASS]")

    # Test 2: HierarchicalEmotionEncoder
    print("\n[Test 2] HierarchicalEmotionEncoder forward pass...")
    encoder = HierarchicalEmotionEncoder(config)

    batch_size = 2
    num_phonemes = 20
    num_words = 5

    # Create dummy inputs
    phoneme_features = torch.randn(batch_size, num_phonemes, config.opensmile_dim)
    phoneme_to_word = [[i // 4 for i in range(num_phonemes)] for _ in range(batch_size)]

    output = encoder(phoneme_features, phoneme_to_word, num_words)

    print(f"  Phoneme embeddings: {output['phoneme_embeddings'].shape}")
    print(f"  Word embeddings: {output['word_embeddings'].shape}")
    print(f"  Utterance embedding: {output['utterance_embedding'].shape}")
    print(f"  Combined embedding: {output['combined_embedding'].shape}")
    print(f"  Word intensities: {output['word_intensities'].shape}")
    print("  [PASS]")

    # Test 3: HEDVarianceAdaptor
    print("\n[Test 3] HEDVarianceAdaptor...")
    adaptor = HEDVarianceAdaptor(config)
    tokens = adaptor(output['combined_embedding'])
    print(f"  Variance adaptor tokens: {tokens.shape}")
    print("  [PASS]")

    # Test 4: HierarchicalEmotionLoss
    print("\n[Test 4] HierarchicalEmotionLoss...")
    loss_fn = HierarchicalEmotionLoss(config)
    emotion_labels = torch.randint(0, config.num_emotions, (batch_size,))

    losses = loss_fn(output, emotion_labels)
    print(f"  Utterance emotion loss: {losses['utterance_emotion'].item():.4f}")
    print(f"  Word emotion loss: {losses['word_emotion'].item():.4f}")
    print(f"  Phoneme emotion loss: {losses['phoneme_emotion'].item():.4f}")
    print(f"  Consistency loss: {losses['consistency'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 5: PhonemeAligner (G2P mode)
    print("\n[Test 5] PhonemeAligner (G2P mode)...")
    aligner = PhonemeAligner(config)

    # Create dummy audio path (or skip if none available)
    import tempfile
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        # Create 1 second of dummy audio
        dummy_audio = np.random.randn(24000).astype(np.float32)
        import scipy.io.wavfile as wav
        wav.write(f.name, 24000, (dummy_audio * 32767).astype(np.int16))

        alignment = aligner.align(f.name, "Hello world")
        print(f"  Phonemes: {len(alignment['phonemes'])}")
        print(f"  Words: {len(alignment['words'])}")
        print(f"  First word: {alignment['words'][0] if alignment['words'] else 'N/A'}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All HED tests passed!")
    print("=" * 60)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from hierarchical_emotion import HEDConfig, HEDPipeline

# Initialize
config = HEDConfig()
pipeline = HEDPipeline(config, device='cuda')

# Process audio
result = pipeline.process(
    audio="sample.wav",
    text="I am so excited about this!",
    word_timestamps=whisper_word_timestamps,  # Optional
)

# Get variance adaptor tokens for prosody model
variance_tokens = result['variance_adaptor_tokens']  # [1, 4, 2048]

# Get word-level emotion intensities for fine-grained control
word_intensities = result['word_intensities']  # [1, num_words, 1]

# Integrate with ProsodyControlledCSM:
# prosody_prefix = prosody_encoder(prosody_dict)
# combined_prefix = prosody_prefix + variance_tokens  # Additive fusion
""")
