"""
Voice Clone Pipeline - Scientific Voice Quality Evaluation

Evaluates voice clone quality using multiple metrics:
1. Speaker Similarity (cosine similarity of speaker embeddings)
2. Prosody Metrics (F0 correlation, speaking rate, pause patterns, pitch range)
3. Mel Cepstral Distortion (MCD) - lower is better
4. Word Error Rate (WER) - optional, requires Whisper

Usage:
    python evaluate.py \
        --model ../models/checkpoints/voice_deepseek_v1/final.pt \
        --reference ../data/labeled/sample_001.wav \
        --text "Hello, this is a test of my cloned voice." \
        --output evaluation_report.json
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import torchaudio
import librosa
import parselmouth
from parselmouth.praat import call
from scipy import stats
from scipy.spatial.distance import cosine
from scipy.ndimage import gaussian_filter1d

# Suppress warnings
warnings.filterwarnings("ignore")


# ============== Data Classes ==============

@dataclass
class SpeakerSimilarityResult:
    """Speaker embedding similarity metrics."""
    cosine_similarity: float  # 0-1, higher is better
    euclidean_distance: float  # Lower is better
    embedding_model: str
    reference_embedding_norm: float
    generated_embedding_norm: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProsodyMetrics:
    """Prosody comparison metrics between reference and generated audio."""
    # F0 (Pitch) metrics
    f0_correlation: float  # Pearson correlation, -1 to 1
    f0_rmse: float  # Root mean square error in Hz
    f0_mean_diff: float  # Difference in mean pitch (Hz)
    f0_range_ratio: float  # Ratio of pitch ranges (1.0 = perfect match)
    f0_std_ratio: float  # Ratio of pitch standard deviations

    # Speaking rate metrics
    speaking_rate_ref: float  # Syllables per second
    speaking_rate_gen: float
    speaking_rate_ratio: float  # 1.0 = perfect match

    # Pause pattern metrics
    pause_count_ref: int
    pause_count_gen: int
    pause_total_ratio: float  # Ratio of total pause durations

    # Duration metrics
    duration_ref: float
    duration_gen: float
    duration_ratio: float

    # Overall prosody score (0-1)
    prosody_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MelCepstralDistortion:
    """Mel Cepstral Distortion metrics."""
    mcd: float  # MCD in dB, lower is better
    mcd_normalized: float  # MCD normalized by duration
    frame_count_ref: int
    frame_count_gen: int

    # Quality interpretation
    quality_category: str  # excellent, good, fair, poor

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WordErrorRate:
    """Word Error Rate metrics."""
    wer: float  # 0-1, lower is better
    word_count_target: int
    word_count_transcribed: int
    substitutions: int
    insertions: int
    deletions: int
    target_text: str
    transcribed_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report combining all metrics."""
    # Individual metrics
    speaker_similarity: Optional[SpeakerSimilarityResult]
    prosody: Optional[ProsodyMetrics]
    mcd: Optional[MelCepstralDistortion]
    wer: Optional[WordErrorRate]

    # Overall scores (0-100)
    overall_score: float
    speaker_score: float  # 0-100
    prosody_score_normalized: float  # 0-100
    mcd_score: float  # 0-100, derived from MCD
    intelligibility_score: float  # 0-100, derived from WER

    # Metadata
    model_path: str
    reference_path: str
    generated_path: str
    target_text: str
    evaluation_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "speaker_similarity": self.speaker_similarity.to_dict() if self.speaker_similarity else None,
            "prosody": self.prosody.to_dict() if self.prosody else None,
            "mcd": self.mcd.to_dict() if self.mcd else None,
            "wer": self.wer.to_dict() if self.wer else None,
            "overall_score": self.overall_score,
            "speaker_score": self.speaker_score,
            "prosody_score_normalized": self.prosody_score_normalized,
            "mcd_score": self.mcd_score,
            "intelligibility_score": self.intelligibility_score,
            "model_path": self.model_path,
            "reference_path": self.reference_path,
            "generated_path": self.generated_path,
            "target_text": self.target_text,
            "evaluation_timestamp": self.evaluation_timestamp,
        }
        return result


# ============== Speaker Similarity ==============

class SpeakerSimilarityEvaluator:
    """
    Computes speaker similarity using speaker embeddings.
    Uses SpeechBrain's ECAPA-TDNN model (state-of-the-art speaker recognition).
    Falls back to Resemblyzer if SpeechBrain is unavailable.
    """

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self.model = None
        self.model_name = None
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _load_model(self):
        """Load speaker embedding model (try SpeechBrain first, fallback to Resemblyzer)."""
        # Try SpeechBrain ECAPA-TDNN
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/speechbrain_spkrec",
                run_opts={"device": self.device}
            )
            self.model_name = "speechbrain-ecapa-tdnn"
            print(f"Loaded SpeechBrain ECAPA-TDNN on {self.device}")
            return
        except ImportError:
            print("SpeechBrain not available, trying Resemblyzer...")
        except Exception as e:
            print(f"SpeechBrain failed: {e}, trying Resemblyzer...")

        # Fallback to Resemblyzer
        try:
            from resemblyzer import VoiceEncoder
            self.model = VoiceEncoder(device=self.device)
            self.model_name = "resemblyzer"
            print(f"Loaded Resemblyzer on {self.device}")
            return
        except ImportError:
            print("Resemblyzer not available")
        except Exception as e:
            print(f"Resemblyzer failed: {e}")

        raise RuntimeError(
            "No speaker embedding model available. "
            "Install speechbrain or resemblyzer:\n"
            "  pip install speechbrain\n"
            "  pip install resemblyzer"
        )

    def _extract_embedding_speechbrain(self, audio_path: str) -> np.ndarray:
        """Extract embedding using SpeechBrain."""
        signal = self.model.load_audio(audio_path)
        embedding = self.model.encode_batch(signal.unsqueeze(0))
        return embedding.squeeze().cpu().numpy()

    def _extract_embedding_resemblyzer(self, audio_path: str) -> np.ndarray:
        """Extract embedding using Resemblyzer."""
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(audio_path)
        embedding = self.model.embed_utterance(wav)
        return embedding

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract speaker embedding from audio file."""
        if self.model_name == "speechbrain-ecapa-tdnn":
            return self._extract_embedding_speechbrain(audio_path)
        elif self.model_name == "resemblyzer":
            return self._extract_embedding_resemblyzer(audio_path)
        else:
            raise RuntimeError("No model loaded")

    def evaluate(self, reference_path: str, generated_path: str) -> SpeakerSimilarityResult:
        """
        Compute speaker similarity between reference and generated audio.

        Returns:
            SpeakerSimilarityResult with cosine similarity (0-1, higher is better)
        """
        # Extract embeddings
        ref_embedding = self.extract_embedding(reference_path)
        gen_embedding = self.extract_embedding(generated_path)

        # Normalize embeddings
        ref_norm = np.linalg.norm(ref_embedding)
        gen_norm = np.linalg.norm(gen_embedding)

        ref_normalized = ref_embedding / (ref_norm + 1e-10)
        gen_normalized = gen_embedding / (gen_norm + 1e-10)

        # Cosine similarity (1 - cosine distance)
        cos_sim = 1 - cosine(ref_normalized, gen_normalized)

        # Euclidean distance
        euclidean_dist = np.linalg.norm(ref_embedding - gen_embedding)

        return SpeakerSimilarityResult(
            cosine_similarity=float(cos_sim),
            euclidean_distance=float(euclidean_dist),
            embedding_model=self.model_name,
            reference_embedding_norm=float(ref_norm),
            generated_embedding_norm=float(gen_norm),
        )


# ============== Prosody Evaluation ==============

class ProsodyEvaluator:
    """
    Evaluates prosody similarity between reference and generated audio.
    Uses Parselmouth (Praat) for F0 extraction and librosa for rhythm analysis.
    """

    def __init__(self, min_pitch: float = 75, max_pitch: float = 500, sample_rate: int = 24000):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.sample_rate = sample_rate

    def _extract_f0(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 contour from audio file."""
        sound = parselmouth.Sound(audio_path)
        pitch = call(sound, "To Pitch", 0.0, self.min_pitch, self.max_pitch)

        times = []
        values = []

        for i in range(pitch.n_frames):
            t = pitch.get_time_from_frame_number(i + 1)
            f0 = pitch.get_value_in_frame(i + 1)
            times.append(t)
            values.append(f0 if not np.isnan(f0) else 0.0)

        return np.array(times), np.array(values)

    def _extract_rhythm_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract rhythm features from audio."""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(y) / sr

        # Detect onsets (syllable nuclei approximation)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
        syllable_count = max(1, len(onsets))

        # Detect pauses using RMS energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_threshold = np.mean(rms) * 0.1

        hop_duration = 512 / sr
        is_pause = rms < rms_threshold

        pause_regions = []
        in_pause = False
        pause_start = 0

        for i, is_p in enumerate(is_pause):
            if is_p and not in_pause:
                pause_start = i * hop_duration
                in_pause = True
            elif not is_p and in_pause:
                pause_end = i * hop_duration
                if pause_end - pause_start > 0.1:  # Minimum 100ms pause
                    pause_regions.append((pause_start, pause_end))
                in_pause = False

        pause_count = len(pause_regions)
        pause_total = sum(end - start for start, end in pause_regions)

        speaking_rate = syllable_count / duration if duration > 0 else 0

        return {
            "duration": duration,
            "syllable_count": syllable_count,
            "speaking_rate": speaking_rate,
            "pause_count": pause_count,
            "pause_total": pause_total,
        }

    def _align_f0_contours(self, times1: np.ndarray, values1: np.ndarray,
                           times2: np.ndarray, values2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align two F0 contours using DTW-like interpolation."""
        # Use the longer duration as reference
        max_time = max(times1[-1] if len(times1) > 0 else 0,
                       times2[-1] if len(times2) > 0 else 0)

        if max_time == 0:
            return np.array([0]), np.array([0])

        # Create common time grid
        n_points = 100
        common_times = np.linspace(0, max_time, n_points)

        # Interpolate both contours to common grid
        if len(times1) > 1 and len(values1) > 1:
            voiced1 = values1 > 0
            if np.any(voiced1):
                interp1 = np.interp(common_times, times1[voiced1], values1[voiced1], left=0, right=0)
            else:
                interp1 = np.zeros_like(common_times)
        else:
            interp1 = np.zeros_like(common_times)

        if len(times2) > 1 and len(values2) > 1:
            voiced2 = values2 > 0
            if np.any(voiced2):
                interp2 = np.interp(common_times, times2[voiced2], values2[voiced2], left=0, right=0)
            else:
                interp2 = np.zeros_like(common_times)
        else:
            interp2 = np.zeros_like(common_times)

        return interp1, interp2

    def evaluate(self, reference_path: str, generated_path: str) -> ProsodyMetrics:
        """
        Compare prosody between reference and generated audio.

        Returns:
            ProsodyMetrics with detailed prosody comparison
        """
        # Extract F0 contours
        ref_times, ref_f0 = self._extract_f0(reference_path)
        gen_times, gen_f0 = self._extract_f0(generated_path)

        # Extract rhythm features
        ref_rhythm = self._extract_rhythm_features(reference_path)
        gen_rhythm = self._extract_rhythm_features(generated_path)

        # Align F0 contours
        ref_f0_aligned, gen_f0_aligned = self._align_f0_contours(
            ref_times, ref_f0, gen_times, gen_f0
        )

        # Get voiced regions (F0 > 0)
        voiced_mask = (ref_f0_aligned > 0) & (gen_f0_aligned > 0)

        # F0 correlation
        if np.sum(voiced_mask) > 2:
            ref_voiced = ref_f0_aligned[voiced_mask]
            gen_voiced = gen_f0_aligned[voiced_mask]
            f0_correlation, _ = stats.pearsonr(ref_voiced, gen_voiced)
            f0_rmse = np.sqrt(np.mean((ref_voiced - gen_voiced) ** 2))
        else:
            f0_correlation = 0.0
            f0_rmse = float('inf')

        # F0 statistics
        ref_f0_voiced = ref_f0[ref_f0 > 0] if len(ref_f0) > 0 else np.array([0])
        gen_f0_voiced = gen_f0[gen_f0 > 0] if len(gen_f0) > 0 else np.array([0])

        ref_f0_mean = np.mean(ref_f0_voiced) if len(ref_f0_voiced) > 0 else 0
        gen_f0_mean = np.mean(gen_f0_voiced) if len(gen_f0_voiced) > 0 else 0

        ref_f0_std = np.std(ref_f0_voiced) if len(ref_f0_voiced) > 0 else 0
        gen_f0_std = np.std(gen_f0_voiced) if len(gen_f0_voiced) > 0 else 0

        ref_f0_range = np.ptp(ref_f0_voiced) if len(ref_f0_voiced) > 0 else 0
        gen_f0_range = np.ptp(gen_f0_voiced) if len(gen_f0_voiced) > 0 else 0

        f0_mean_diff = abs(ref_f0_mean - gen_f0_mean)
        f0_range_ratio = gen_f0_range / (ref_f0_range + 1e-10)
        f0_std_ratio = gen_f0_std / (ref_f0_std + 1e-10)

        # Speaking rate ratio
        speaking_rate_ratio = gen_rhythm["speaking_rate"] / (ref_rhythm["speaking_rate"] + 1e-10)

        # Pause pattern ratio
        pause_total_ratio = gen_rhythm["pause_total"] / (ref_rhythm["pause_total"] + 1e-10) if ref_rhythm["pause_total"] > 0 else 1.0

        # Duration ratio
        duration_ratio = gen_rhythm["duration"] / (ref_rhythm["duration"] + 1e-10)

        # Calculate overall prosody score (0-1)
        # Weighted combination of normalized metrics
        scores = []

        # F0 correlation (already 0-1 range, but can be negative)
        f0_corr_score = max(0, (f0_correlation + 1) / 2)  # Map -1,1 to 0,1
        scores.append(f0_corr_score * 0.3)  # 30% weight

        # F0 RMSE score (lower is better, normalize to 0-1)
        f0_rmse_score = np.exp(-f0_rmse / 50) if f0_rmse != float('inf') else 0  # 50Hz = e^-1 score
        scores.append(f0_rmse_score * 0.2)  # 20% weight

        # Speaking rate match (1.0 = perfect, penalize deviation)
        rate_score = np.exp(-abs(speaking_rate_ratio - 1) * 2)
        scores.append(rate_score * 0.2)  # 20% weight

        # Pitch range match
        range_score = np.exp(-abs(f0_range_ratio - 1) * 2)
        scores.append(range_score * 0.15)  # 15% weight

        # Duration match
        duration_score = np.exp(-abs(duration_ratio - 1) * 2)
        scores.append(duration_score * 0.15)  # 15% weight

        prosody_score = sum(scores)

        return ProsodyMetrics(
            f0_correlation=float(f0_correlation) if not np.isnan(f0_correlation) else 0.0,
            f0_rmse=float(f0_rmse) if f0_rmse != float('inf') else 999.0,
            f0_mean_diff=float(f0_mean_diff),
            f0_range_ratio=float(f0_range_ratio),
            f0_std_ratio=float(f0_std_ratio),
            speaking_rate_ref=float(ref_rhythm["speaking_rate"]),
            speaking_rate_gen=float(gen_rhythm["speaking_rate"]),
            speaking_rate_ratio=float(speaking_rate_ratio),
            pause_count_ref=ref_rhythm["pause_count"],
            pause_count_gen=gen_rhythm["pause_count"],
            pause_total_ratio=float(pause_total_ratio),
            duration_ref=float(ref_rhythm["duration"]),
            duration_gen=float(gen_rhythm["duration"]),
            duration_ratio=float(duration_ratio),
            prosody_score=float(prosody_score),
        )


# ============== Mel Cepstral Distortion ==============

class MCDEvaluator:
    """
    Computes Mel Cepstral Distortion (MCD) between reference and generated audio.
    MCD is a standard metric for evaluating speech synthesis quality.
    Lower MCD = better quality.
    """

    def __init__(self, sample_rate: int = 24000, n_mfcc: int = 13, n_mels: int = 80):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def _extract_mfcc(self, audio_path: str) -> np.ndarray:
        """Extract MFCC features from audio."""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Extract MFCCs (excluding C0)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=self.n_mfcc + 1,  # +1 to exclude C0
            n_mels=self.n_mels,
            hop_length=256,
            n_fft=1024,
        )

        # Exclude C0 (energy coefficient)
        mfcc = mfcc[1:]  # Shape: (n_mfcc, frames)

        return mfcc.T  # Shape: (frames, n_mfcc)

    def _align_mfcc(self, mfcc1: np.ndarray, mfcc2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align two MFCC sequences using simple interpolation."""
        len1, len2 = len(mfcc1), len(mfcc2)

        # Interpolate to common length
        common_len = max(len1, len2)

        if len1 != common_len:
            indices = np.linspace(0, len1 - 1, common_len)
            mfcc1_aligned = np.array([
                np.interp(indices, np.arange(len1), mfcc1[:, i])
                for i in range(mfcc1.shape[1])
            ]).T
        else:
            mfcc1_aligned = mfcc1

        if len2 != common_len:
            indices = np.linspace(0, len2 - 1, common_len)
            mfcc2_aligned = np.array([
                np.interp(indices, np.arange(len2), mfcc2[:, i])
                for i in range(mfcc2.shape[1])
            ]).T
        else:
            mfcc2_aligned = mfcc2

        return mfcc1_aligned, mfcc2_aligned

    def evaluate(self, reference_path: str, generated_path: str) -> MelCepstralDistortion:
        """
        Compute MCD between reference and generated audio.

        MCD formula: (10 / ln(10)) * sqrt(2 * sum((c1 - c2)^2))

        Returns:
            MelCepstralDistortion with MCD value in dB
        """
        # Extract MFCCs
        ref_mfcc = self._extract_mfcc(reference_path)
        gen_mfcc = self._extract_mfcc(generated_path)

        # Align sequences
        ref_aligned, gen_aligned = self._align_mfcc(ref_mfcc, gen_mfcc)

        # Calculate MCD
        # MCD = (10 / ln(10)) * sqrt(2 * sum((c1_i - c2_i)^2))
        # Note: Skip first coefficient (c0, energy) as per standard practice
        diff = ref_aligned[:, 1:] - gen_aligned[:, 1:]  # Exclude c0
        frame_distances = np.sqrt(2 * np.sum(diff ** 2, axis=1))
        mcd = (10.0 / np.log(10)) * np.mean(frame_distances)

        # Typical MCD values are 4-8 dB for good synthesis
        # If values are way off, it may indicate a scaling issue
        # Clamp to reasonable range for score calculation
        mcd = min(mcd, 20.0)  # Cap at 20 dB for scoring purposes

        # Normalized MCD (per frame)
        mcd_normalized = mcd

        # Quality interpretation based on typical MCD ranges
        # < 4 dB: Excellent (nearly indistinguishable)
        # 4-6 dB: Good (high quality synthesis)
        # 6-8 dB: Fair (acceptable synthesis)
        # > 8 dB: Poor (noticeable differences)
        if mcd < 4:
            quality = "excellent"
        elif mcd < 6:
            quality = "good"
        elif mcd < 8:
            quality = "fair"
        else:
            quality = "poor"

        return MelCepstralDistortion(
            mcd=float(mcd),
            mcd_normalized=float(mcd_normalized),
            frame_count_ref=len(ref_mfcc),
            frame_count_gen=len(gen_mfcc),
            quality_category=quality,
        )


# ============== Word Error Rate ==============

class WEREvaluator:
    """
    Computes Word Error Rate by transcribing generated audio
    and comparing to target text.
    """

    def __init__(self, model_name: str = "base", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None

    def _load_model(self):
        """Lazy load Whisper model."""
        if self.model is None:
            import whisper
            print(f"Loading Whisper {self.model_name}...")
            self.model = whisper.load_model(self.model_name)
            print("Whisper loaded!")

    def _calculate_wer(self, reference: str, hypothesis: str) -> Tuple[float, int, int, int]:
        """
        Calculate WER using dynamic programming (Levenshtein distance).

        Returns:
            (wer, substitutions, insertions, deletions)
        """
        # Normalize and tokenize
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        # Handle edge cases
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0, 0, len(hyp_words), 0

        # Dynamic programming for edit distance
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)

        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        # Backtrack to find S, I, D
        i, j = len(ref_words), len(hyp_words)
        substitutions = insertions = deletions = 0

        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif j > 0 and d[i][j] == d[i][j - 1] + 1:
                insertions += 1
                j -= 1
            else:
                deletions += 1
                i -= 1

        wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)

        return wer, substitutions, insertions, deletions

    def evaluate(self, generated_path: str, target_text: str) -> WordErrorRate:
        """
        Transcribe generated audio and compute WER against target text.

        Returns:
            WordErrorRate with WER value (0-1, lower is better)
        """
        self._load_model()

        # Transcribe generated audio
        result = self.model.transcribe(generated_path, language="en")
        transcribed_text = result["text"].strip()

        # Calculate WER
        wer, subs, ins, dels = self._calculate_wer(target_text, transcribed_text)

        return WordErrorRate(
            wer=float(min(wer, 1.0)),  # Cap at 1.0
            word_count_target=len(target_text.split()),
            word_count_transcribed=len(transcribed_text.split()),
            substitutions=subs,
            insertions=ins,
            deletions=dels,
            target_text=target_text,
            transcribed_text=transcribed_text,
        )


# ============== Complete Evaluator ==============

class VoiceCloneEvaluator:
    """
    Complete voice clone evaluation combining all metrics.
    """

    def __init__(
        self,
        use_speaker_similarity: bool = True,
        use_prosody: bool = True,
        use_mcd: bool = True,
        use_wer: bool = True,
        whisper_model: str = "base",
        device: str = "auto",
    ):
        self.use_speaker_similarity = use_speaker_similarity
        self.use_prosody = use_prosody
        self.use_mcd = use_mcd
        self.use_wer = use_wer

        # Initialize evaluators
        self.speaker_evaluator = None
        self.prosody_evaluator = None
        self.mcd_evaluator = None
        self.wer_evaluator = None

        if use_speaker_similarity:
            try:
                self.speaker_evaluator = SpeakerSimilarityEvaluator(device=device)
            except Exception as e:
                print(f"Warning: Could not initialize speaker similarity: {e}")

        if use_prosody:
            self.prosody_evaluator = ProsodyEvaluator()

        if use_mcd:
            self.mcd_evaluator = MCDEvaluator()

        if use_wer:
            try:
                self.wer_evaluator = WEREvaluator(model_name=whisper_model, device=device)
            except Exception as e:
                print(f"Warning: Could not initialize WER evaluator: {e}")

    def _mcd_to_score(self, mcd: float) -> float:
        """Convert MCD to 0-100 score (lower MCD = higher score)."""
        # MCD typically ranges from 2-12 dB
        # Map to 0-100 using exponential decay
        # MCD=2 -> ~100, MCD=6 -> ~50, MCD=10 -> ~10
        return float(max(0, min(100, 100 * np.exp(-mcd / 5))))

    def _wer_to_score(self, wer: float) -> float:
        """Convert WER to 0-100 score (lower WER = higher score)."""
        # WER 0 = 100, WER 0.5 = 50, WER 1 = 0
        return float(max(0, min(100, 100 * (1 - wer))))

    def evaluate(
        self,
        reference_path: str,
        generated_path: str,
        target_text: str,
        model_path: str = "",
    ) -> EvaluationReport:
        """
        Run complete evaluation on generated audio.

        Args:
            reference_path: Path to reference audio (user's voice)
            generated_path: Path to generated audio
            target_text: Text that was synthesized
            model_path: Path to model used for generation

        Returns:
            EvaluationReport with all metrics
        """
        from datetime import datetime

        print("Running voice clone evaluation...")

        # Speaker similarity
        speaker_result = None
        speaker_score = 0.0
        if self.speaker_evaluator:
            print("  Computing speaker similarity...")
            try:
                speaker_result = self.speaker_evaluator.evaluate(reference_path, generated_path)
                speaker_score = speaker_result.cosine_similarity * 100
            except Exception as e:
                print(f"  Speaker similarity failed: {e}")

        # Prosody
        prosody_result = None
        prosody_score = 0.0
        if self.prosody_evaluator:
            print("  Computing prosody metrics...")
            try:
                prosody_result = self.prosody_evaluator.evaluate(reference_path, generated_path)
                prosody_score = prosody_result.prosody_score * 100
            except Exception as e:
                print(f"  Prosody evaluation failed: {e}")

        # MCD
        mcd_result = None
        mcd_score = 0.0
        if self.mcd_evaluator:
            print("  Computing MCD...")
            try:
                mcd_result = self.mcd_evaluator.evaluate(reference_path, generated_path)
                mcd_score = self._mcd_to_score(mcd_result.mcd)
            except Exception as e:
                print(f"  MCD evaluation failed: {e}")

        # WER
        wer_result = None
        intelligibility_score = 100.0  # Default if not computed
        if self.wer_evaluator and target_text:
            print("  Computing WER...")
            try:
                wer_result = self.wer_evaluator.evaluate(generated_path, target_text)
                intelligibility_score = self._wer_to_score(wer_result.wer)
            except Exception as e:
                print(f"  WER evaluation failed: {e}")

        # Overall score (weighted average)
        weights = {"speaker": 0.35, "prosody": 0.25, "mcd": 0.25, "intelligibility": 0.15}
        scores = {
            "speaker": speaker_score,
            "prosody": prosody_score,
            "mcd": mcd_score,
            "intelligibility": intelligibility_score,
        }

        # Only include metrics that were computed
        valid_weights = {}
        valid_scores = {}

        if speaker_result:
            valid_weights["speaker"] = weights["speaker"]
            valid_scores["speaker"] = scores["speaker"]
        if prosody_result:
            valid_weights["prosody"] = weights["prosody"]
            valid_scores["prosody"] = scores["prosody"]
        if mcd_result:
            valid_weights["mcd"] = weights["mcd"]
            valid_scores["mcd"] = scores["mcd"]
        if wer_result:
            valid_weights["intelligibility"] = weights["intelligibility"]
            valid_scores["intelligibility"] = scores["intelligibility"]

        # Normalize weights and compute overall score
        total_weight = sum(valid_weights.values())
        if total_weight > 0:
            overall_score = sum(
                valid_scores[k] * valid_weights[k] / total_weight
                for k in valid_weights
            )
        else:
            overall_score = 0.0

        return EvaluationReport(
            speaker_similarity=speaker_result,
            prosody=prosody_result,
            mcd=mcd_result,
            wer=wer_result,
            overall_score=float(overall_score),
            speaker_score=float(speaker_score),
            prosody_score_normalized=float(prosody_score),
            mcd_score=float(mcd_score),
            intelligibility_score=float(intelligibility_score),
            model_path=model_path,
            reference_path=reference_path,
            generated_path=generated_path,
            target_text=target_text,
            evaluation_timestamp=datetime.now().isoformat(),
        )

    def evaluate_batch(
        self,
        reference_paths: List[str],
        generated_paths: List[str],
        target_texts: List[str],
        model_path: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate multiple audio pairs and compute aggregate statistics.
        """
        reports = []

        for ref, gen, text in zip(reference_paths, generated_paths, target_texts):
            report = self.evaluate(ref, gen, text, model_path)
            reports.append(report)

        # Aggregate statistics
        if not reports:
            return {"error": "No reports generated"}

        def safe_mean(values):
            valid = [v for v in values if v is not None and not np.isnan(v)]
            return float(np.mean(valid)) if valid else 0.0

        def safe_std(values):
            valid = [v for v in values if v is not None and not np.isnan(v)]
            return float(np.std(valid)) if len(valid) > 1 else 0.0

        return {
            "count": len(reports),
            "overall_score": {
                "mean": safe_mean([r.overall_score for r in reports]),
                "std": safe_std([r.overall_score for r in reports]),
            },
            "speaker_score": {
                "mean": safe_mean([r.speaker_score for r in reports]),
                "std": safe_std([r.speaker_score for r in reports]),
            },
            "prosody_score": {
                "mean": safe_mean([r.prosody_score_normalized for r in reports]),
                "std": safe_std([r.prosody_score_normalized for r in reports]),
            },
            "mcd_score": {
                "mean": safe_mean([r.mcd_score for r in reports]),
                "std": safe_std([r.mcd_score for r in reports]),
            },
            "intelligibility_score": {
                "mean": safe_mean([r.intelligibility_score for r in reports]),
                "std": safe_std([r.intelligibility_score for r in reports]),
            },
            "individual_reports": [r.to_dict() for r in reports],
        }


# ============== CLI Interface ==============

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate voice clone quality with multiple metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with existing generated audio
  python evaluate.py --reference ref.wav --generated gen.wav --text "Hello world"

  # Generate audio with model and evaluate
  python evaluate.py --model ../models/checkpoints/voice_v1/best.pt \\
                     --reference ref.wav --text "Hello world"

  # Full evaluation with all metrics
  python evaluate.py --reference ref.wav --generated gen.wav --text "Hello" \\
                     --output report.json --whisper-model large-v3

Score Interpretation:
  - Overall Score (0-100): Weighted average of all metrics
  - Speaker Similarity: Higher is better (>0.8 = excellent)
  - MCD: Lower is better (<4dB = excellent, 4-6 = good, 6-8 = fair, >8 = poor)
  - WER: Lower is better (0 = perfect transcription)
  - Prosody Score: Higher is better (weighted F0 correlation + rhythm match)
        """
    )

    parser.add_argument("--model", "-m", help="Path to model checkpoint (for generation)")
    parser.add_argument("--reference", "-r", required=True, help="Reference audio (user's voice)")
    parser.add_argument("--generated", "-g", help="Generated audio (if not using --model)")
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize/evaluate")
    parser.add_argument("--output", "-o", default="evaluation_report.json", help="Output JSON report")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, mps, cpu)")
    parser.add_argument("--whisper-model", default="base", help="Whisper model for WER (base, small, medium, large-v3)")
    parser.add_argument("--no-speaker", action="store_true", help="Skip speaker similarity")
    parser.add_argument("--no-prosody", action="store_true", help="Skip prosody evaluation")
    parser.add_argument("--no-mcd", action="store_true", help="Skip MCD evaluation")
    parser.add_argument("--no-wer", action="store_true", help="Skip WER evaluation")

    args = parser.parse_args()

    # Validate inputs
    reference_path = Path(args.reference)
    if not reference_path.exists():
        print(f"Error: Reference audio not found: {reference_path}")
        sys.exit(1)

    # Get generated audio path
    if args.generated:
        generated_path = Path(args.generated)
        if not generated_path.exists():
            print(f"Error: Generated audio not found: {generated_path}")
            sys.exit(1)
    elif args.model:
        # Generate audio using model
        print("Generating audio...")
        from generate import VoiceGenerator

        generator = VoiceGenerator(args.model, device=args.device)
        audio = generator.generate(text=args.text)

        generated_path = Path("temp_generated.wav")
        generator.save_audio(audio, str(generated_path))
    else:
        print("Error: Either --generated or --model must be provided")
        sys.exit(1)

    # Create evaluator
    evaluator = VoiceCloneEvaluator(
        use_speaker_similarity=not args.no_speaker,
        use_prosody=not args.no_prosody,
        use_mcd=not args.no_mcd,
        use_wer=not args.no_wer,
        whisper_model=args.whisper_model,
        device=args.device,
    )

    # Run evaluation
    report = evaluator.evaluate(
        reference_path=str(reference_path),
        generated_path=str(generated_path),
        target_text=args.text,
        model_path=args.model or "",
    )

    # Save report
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\n{'='*60}")
    print("VOICE CLONE EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"\nOverall Score: {report.overall_score:.1f}/100")
    print(f"\nComponent Scores:")
    print(f"  Speaker Similarity: {report.speaker_score:.1f}/100")
    print(f"  Prosody Match:      {report.prosody_score_normalized:.1f}/100")
    print(f"  MCD Score:          {report.mcd_score:.1f}/100")
    print(f"  Intelligibility:    {report.intelligibility_score:.1f}/100")

    if report.mcd:
        print(f"\nMCD Details:")
        print(f"  MCD Value: {report.mcd.mcd:.2f} dB ({report.mcd.quality_category})")

    if report.speaker_similarity:
        print(f"\nSpeaker Similarity Details:")
        print(f"  Cosine Similarity: {report.speaker_similarity.cosine_similarity:.4f}")
        print(f"  Model: {report.speaker_similarity.embedding_model}")

    if report.prosody:
        print(f"\nProsody Details:")
        print(f"  F0 Correlation: {report.prosody.f0_correlation:.3f}")
        print(f"  F0 RMSE: {report.prosody.f0_rmse:.1f} Hz")
        print(f"  Speaking Rate Ratio: {report.prosody.speaking_rate_ratio:.2f}")

    if report.wer:
        print(f"\nWER Details:")
        print(f"  WER: {report.wer.wer:.2%}")
        print(f"  Target: {report.wer.target_text}")
        print(f"  Transcribed: {report.wer.transcribed_text}")

    print(f"\nReport saved to: {output_path}")

    # Cleanup temp file if we generated it
    if not args.generated and args.model:
        generated_path.unlink()


if __name__ == "__main__":
    main()
