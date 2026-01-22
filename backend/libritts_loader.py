"""
LibriTTS-R Annotated Dataset Loader

Provides utilities for loading and querying the LibriTTS-R annotated dataset
for prosody comparison and testing purposes.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

import pyarrow.ipc as ipc


# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LIBRITTS_DIR = DATA_DIR / "libritts_annotated" / "train_clean_100"
CACHE_DIR = DATA_DIR / "cache" / "libritts_audio"


@dataclass
class LibriTTSSample:
    """A sample from the LibriTTS annotated dataset."""
    id: str
    text: str
    speaker_id: str
    chapter_id: str
    original_path: str

    # Prosody annotations
    speaking_rate: str  # categorical
    gender: str
    pitch: str  # categorical
    speech_monotony: str
    utterance_pitch_mean: float
    utterance_pitch_std: float

    # Quality metrics
    snr: float
    c50: float
    noise: str
    reverberation: str

    # Description
    text_description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LibriTTSLoader:
    """Loader for LibriTTS-R annotated dataset."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or LIBRITTS_DIR
        self.arrow_path = self.data_dir / "data-00000-of-00001.arrow"
        self._table = None
        self._samples_cache: Dict[str, LibriTTSSample] = {}

    def _load_table(self):
        """Lazy load the arrow table."""
        if self._table is None:
            if not self.arrow_path.exists():
                raise FileNotFoundError(f"LibriTTS data not found at {self.arrow_path}")
            with open(self.arrow_path, "rb") as f:
                reader = ipc.open_stream(f)
                self._table = reader.read_all()
        return self._table

    @property
    def total_count(self) -> int:
        """Total number of samples in the dataset."""
        table = self._load_table()
        return table.num_rows

    def get_sample(self, sample_id: str) -> Optional[LibriTTSSample]:
        """Get a single sample by ID."""
        if sample_id in self._samples_cache:
            return self._samples_cache[sample_id]

        table = self._load_table()

        # Search for the sample ID
        ids = table.column("id").to_pylist()
        try:
            idx = ids.index(sample_id)
        except ValueError:
            return None

        sample = self._row_to_sample(table, idx)
        self._samples_cache[sample_id] = sample
        return sample

    def list_samples(
        self,
        limit: int = 50,
        offset: int = 0,
        gender: Optional[str] = None,
        pitch_category: Optional[str] = None,
        speaking_rate: Optional[str] = None,
        speaker_id: Optional[str] = None,
        search_text: Optional[str] = None,
    ) -> Tuple[List[LibriTTSSample], int]:
        """
        List samples with pagination and filtering.

        Returns (samples, total_matching_count).
        """
        table = self._load_table()

        # Get all data as lists for filtering
        ids = table.column("id").to_pylist()
        texts = table.column("text").to_pylist()
        genders = table.column("gender").to_pylist()
        pitches = table.column("pitch").to_pylist()
        rates = table.column("speaking_rate").to_pylist()
        speakers = [str(x) for x in table.column("speaker_id").to_pylist()]

        # Apply filters
        indices = list(range(len(ids)))

        if gender:
            indices = [i for i in indices if genders[i] == gender]

        if pitch_category:
            indices = [i for i in indices if pitches[i] == pitch_category]

        if speaking_rate:
            indices = [i for i in indices if rates[i] == speaking_rate]

        if speaker_id:
            indices = [i for i in indices if speakers[i] == speaker_id]

        if search_text:
            search_lower = search_text.lower()
            indices = [i for i in indices if search_lower in texts[i].lower()]

        total = len(indices)

        # Apply pagination
        paginated_indices = indices[offset:offset + limit]

        # Convert to samples
        samples = [self._row_to_sample(table, i) for i in paginated_indices]

        return samples, total

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        table = self._load_table()

        genders = table.column("gender").to_pylist()
        pitches = table.column("pitch").to_pylist()
        rates = table.column("speaking_rate").to_pylist()
        speakers = [str(x) for x in table.column("speaker_id").to_pylist()]

        # Count distributions
        gender_counts = {}
        for g in genders:
            gender_counts[g] = gender_counts.get(g, 0) + 1

        pitch_counts = {}
        for p in pitches:
            pitch_counts[p] = pitch_counts.get(p, 0) + 1

        rate_counts = {}
        for r in rates:
            rate_counts[r] = rate_counts.get(r, 0) + 1

        return {
            "total_samples": table.num_rows,
            "unique_speakers": len(set(speakers)),
            "gender_distribution": gender_counts,
            "pitch_distribution": pitch_counts,
            "speaking_rate_distribution": rate_counts,
        }

    def _row_to_sample(self, table, idx: int) -> LibriTTSSample:
        """Convert a table row to a LibriTTSSample."""
        return LibriTTSSample(
            id=table.column("id")[idx].as_py(),
            text=table.column("text")[idx].as_py(),
            speaker_id=str(table.column("speaker_id")[idx].as_py()),
            chapter_id=str(table.column("chapter_id")[idx].as_py()),
            original_path=table.column("path")[idx].as_py(),
            speaking_rate=table.column("speaking_rate")[idx].as_py(),
            gender=table.column("gender")[idx].as_py(),
            pitch=table.column("pitch")[idx].as_py(),
            speech_monotony=table.column("speech_monotony")[idx].as_py(),
            utterance_pitch_mean=float(table.column("utterance_pitch_mean")[idx].as_py()),
            utterance_pitch_std=float(table.column("utterance_pitch_std")[idx].as_py()),
            snr=float(table.column("snr")[idx].as_py()),
            c50=float(table.column("c50")[idx].as_py()),
            noise=table.column("noise")[idx].as_py(),
            reverberation=table.column("reverberation")[idx].as_py(),
            text_description=table.column("text_description")[idx].as_py(),
        )


# ============== Mapping Functions ==============

def pitch_hz_to_category(pitch_hz: float, gender: str) -> str:
    """Map pitch in Hz to LibriTTS categorical labels.

    LibriTTS uses 7 pitch categories including "slightly low pitch".
    Note: LibriTTS annotations are speaker-normalized, so Hz-based mapping
    is an approximation. These thresholds are optimized for best average
    accuracy across diverse speakers.

    Female typical F0: 165-255 Hz
    Male typical F0: 85-180 Hz
    """
    if gender.lower() == "female":
        if pitch_hz < 150:
            return "very low pitch"
        elif pitch_hz < 175:
            return "quite low pitch"
        elif pitch_hz < 195:
            return "slightly low pitch"
        elif pitch_hz < 225:
            return "moderate pitch"
        elif pitch_hz < 255:
            return "slightly high pitch"
        elif pitch_hz < 285:
            return "quite high pitch"
        else:
            return "very high pitch"
    else:  # male
        if pitch_hz < 95:
            return "very low pitch"
        elif pitch_hz < 108:
            return "quite low pitch"
        elif pitch_hz < 122:
            return "slightly low pitch"
        elif pitch_hz < 145:
            return "moderate pitch"
        elif pitch_hz < 165:
            return "slightly high pitch"
        elif pitch_hz < 185:
            return "quite high pitch"
        else:
            return "very high pitch"


def speaking_rate_sps_to_category(sps: float) -> str:
    """Map syllables per second to LibriTTS categorical labels.

    Thresholds calibrated against LibriTTS-R annotations.
    LibriTTS uses onset-based syllable detection which tends to
    give higher rates than pure acoustic-based methods.
    """
    if sps < 2.5:
        return "very slowly"
    elif sps < 3.2:
        return "quite slowly"
    elif sps < 3.8:
        return "slightly slowly"
    elif sps < 4.5:
        return "moderate speed"
    elif sps < 5.5:
        return "slightly fast"
    elif sps < 6.5:
        return "quite fast"
    else:
        return "very fast"


def infer_gender_from_pitch(pitch_hz: float) -> str:
    """Infer gender from fundamental frequency.

    Based on typical adult F0 ranges:
    - Adult males: 85-180 Hz (modal ~120 Hz)
    - Adult females: 165-255 Hz (modal ~200 Hz)

    The overlap zone is narrower than previously assumed.
    Thresholds tuned against LibriTTS-R speaker demographics.
    """
    if pitch_hz < 150:
        return "male"
    elif pitch_hz > 165:
        return "female"
    else:
        return "uncertain"


# ============== Synthetic Audio Generation ==============

def generate_synthetic_audio(
    sample: LibriTTSSample,
    cache_dir: Optional[Path] = None,
    duration: float = 3.0
) -> Path:
    """
    Generate synthetic audio with known prosody characteristics for testing.
    """
    import soundfile as sf

    cache_dir = cache_dir or (CACHE_DIR / "synthetic")
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / f"{sample.id}_synthetic.wav"

    if audio_path.exists():
        return audio_path

    sr = 24000
    t = np.linspace(0, duration, int(sr * duration))

    f0 = sample.utterance_pitch_mean if sample.utterance_pitch_mean > 0 else 150.0
    f0_std = sample.utterance_pitch_std if sample.utterance_pitch_std > 0 else 20.0

    # Create pitch contour with variation
    pitch_contour = f0 + np.random.normal(0, f0_std * 0.5, len(t))
    pitch_contour = np.clip(pitch_contour, f0 * 0.7, f0 * 1.3)

    # Generate audio with varying pitch
    phase = np.cumsum(2 * np.pi * pitch_contour / sr)
    audio = 0.5 * np.sin(phase)

    # Add harmonics
    audio += 0.3 * np.sin(2 * phase)
    audio += 0.15 * np.sin(3 * phase)
    audio += 0.1 * np.sin(4 * phase)

    # Apply amplitude envelope with pauses
    envelope = np.ones_like(audio)
    if "slow" in sample.speaking_rate.lower():
        n_pauses = 3
    elif "fast" in sample.speaking_rate.lower():
        n_pauses = 1
    else:
        n_pauses = 2

    for _ in range(n_pauses):
        pause_start = np.random.randint(len(audio) // 4, 3 * len(audio) // 4)
        pause_len = int(sr * 0.1)
        envelope[pause_start:min(pause_start + pause_len, len(envelope))] = 0.05

    audio *= envelope
    audio = audio / np.max(np.abs(audio)) * 0.8

    sf.write(str(audio_path), audio, sr)
    return audio_path


# Global instance
_loader: Optional[LibriTTSLoader] = None


def get_loader() -> LibriTTSLoader:
    """Get or create the global LibriTTS loader."""
    global _loader
    if _loader is None:
        _loader = LibriTTSLoader()
    return _loader
