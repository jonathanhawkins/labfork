"""
Voice Clone Pipeline - Multi-Layer Prosody Analyzer
Analyzes audio from 4 perspectives: acoustic, rhythm, semantic, and contour.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torchaudio
import parselmouth
from parselmouth.praat import call
import librosa


@dataclass
class AcousticFeatures:
    """Physical properties of the voice signal."""
    pitch_mean: float
    pitch_min: float
    pitch_max: float
    pitch_range: float
    pitch_std: float
    intensity_mean: float
    intensity_std: float
    jitter: float  # Pitch stability
    shimmer: float  # Amplitude stability
    hnr: float  # Harmonics-to-noise ratio (voice clarity)
    f1_mean: float  # First formant
    f2_mean: float  # Second formant
    f3_mean: float  # Third formant


@dataclass
class RhythmFeatures:
    """Timing and rhythm properties."""
    duration_seconds: float
    speaking_rate: float  # Syllables per second (including pauses)
    articulation_rate: float  # Syllables per second (excluding pauses)
    pause_count: int
    pause_total_duration: float
    pause_mean_duration: float
    speech_to_pause_ratio: float
    syllable_count: int


@dataclass
class SemanticFeatures:
    """AI-interpreted emotional and stylistic features."""
    emotion: str
    emotion_confidence: float
    tone: str
    energy_level: str
    pace_category: str
    emphasis_words: List[str]
    mood: str
    notes: str


@dataclass
class PitchContour:
    """Pitch trajectory over time."""
    times: List[float]
    values: List[float]
    smoothed: List[float]


@dataclass
class ProsodyResult:
    """Complete prosody analysis result."""
    acoustic: AcousticFeatures
    rhythm: RhythmFeatures
    semantic: Optional[SemanticFeatures]
    contour: PitchContour
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "acoustic": asdict(self.acoustic),
            "rhythm": asdict(self.rhythm),
            "semantic": asdict(self.semantic) if self.semantic else None,
            "contour": asdict(self.contour),
        }


class AcousticAnalyzer:
    """
    Extracts acoustic features using Parselmouth (Praat).
    These are the physical properties of the sound wave.
    """
    
    def __init__(self, min_pitch: float = 75, max_pitch: float = 500):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
    
    def analyze(self, audio_path: str) -> AcousticFeatures:
        """Extract acoustic features from audio file."""
        
        # Load with Parselmouth
        sound = parselmouth.Sound(audio_path)
        
        # Extract pitch
        pitch = call(sound, "To Pitch", 0.0, self.min_pitch, self.max_pitch)
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced
        
        if len(pitch_values) == 0:
            pitch_values = np.array([0])
        
        # Extract intensity
        intensity = call(sound, "To Intensity", self.min_pitch, 0.0, True)
        intensity_values = intensity.values.flatten()
        
        # Extract voice quality measures
        point_process = call(sound, "To PointProcess (periodic, cc)", self.min_pitch, self.max_pitch)
        
        try:
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        except:
            jitter = 0.0
        
        try:
            shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except:
            shimmer = 0.0
        
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, self.min_pitch, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
        except:
            hnr = 0.0
        
        # Extract formants
        formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        
        try:
            f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
            f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
            f3 = call(formant, "Get mean", 3, 0, 0, "Hertz")
        except:
            f1, f2, f3 = 500, 1500, 2500
        
        return AcousticFeatures(
            pitch_mean=float(np.mean(pitch_values)),
            pitch_min=float(np.min(pitch_values)),
            pitch_max=float(np.max(pitch_values)),
            pitch_range=float(np.max(pitch_values) - np.min(pitch_values)),
            pitch_std=float(np.std(pitch_values)),
            intensity_mean=float(np.mean(intensity_values)),
            intensity_std=float(np.std(intensity_values)),
            jitter=float(jitter) if not np.isnan(jitter) else 0.0,
            shimmer=float(shimmer) if not np.isnan(shimmer) else 0.0,
            hnr=float(hnr) if not np.isnan(hnr) else 0.0,
            f1_mean=float(f1) if not np.isnan(f1) else 500.0,
            f2_mean=float(f2) if not np.isnan(f2) else 1500.0,
            f3_mean=float(f3) if not np.isnan(f3) else 2500.0,
        )


class RhythmAnalyzer:
    """
    Extracts rhythm and timing features using librosa.
    These capture the temporal patterns of speech.
    """
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
    
    def analyze(self, audio_path: str) -> RhythmFeatures:
        """Extract rhythm features from audio file."""
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(y) / sr
        
        # Detect syllable nuclei using onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
        syllable_count = max(1, len(onsets))
        
        # Detect pauses using RMS energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_threshold = np.mean(rms) * 0.1  # 10% of mean energy
        
        # Find pause regions
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
        pause_mean = pause_total / pause_count if pause_count > 0 else 0
        
        speech_duration = duration - pause_total
        speech_to_pause = speech_duration / pause_total if pause_total > 0 else float('inf')
        
        # Calculate rates
        speaking_rate = syllable_count / duration if duration > 0 else 0
        articulation_rate = syllable_count / speech_duration if speech_duration > 0 else 0
        
        return RhythmFeatures(
            duration_seconds=float(duration),
            speaking_rate=float(speaking_rate),
            articulation_rate=float(articulation_rate),
            pause_count=pause_count,
            pause_total_duration=float(pause_total),
            pause_mean_duration=float(pause_mean),
            speech_to_pause_ratio=float(min(speech_to_pause, 100)),  # Cap for readability
            syllable_count=syllable_count,
        )


class SemanticAnalyzer:
    """
    Uses Qwen2-Audio to interpret emotional and stylistic content.
    This is the "what does it mean" layer.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.processor = None
    
    def load_model(self):
        """Lazy load Qwen2-Audio model."""
        if self.model is None:
            print("Loading Qwen2-Audio...")
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
            
            model_name = "Qwen/Qwen2-Audio-7B-Instruct"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device if self.device != "cpu" else None,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            print("Qwen2-Audio loaded!")
    
    def analyze(self, audio_path: str, transcript: str = "") -> SemanticFeatures:
        """Analyze semantic/emotional content of audio."""
        
        self.load_model()
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Prepare prompt
        prompt = f"""Analyze the prosody of this speech. The transcript is: "{transcript}"

Respond in JSON format with these fields:
- emotion: The primary emotion (one of: neutral, happy, sad, angry, fearful, surprised, disgusted, contemptuous, friendly, excited, thoughtful, concerned, confident, uncertain)
- emotion_confidence: Confidence score 0-1
- tone: The speaking tone (one of: conversational, formal, casual, questioning, declarative, exclamatory, narrative, instructional)
- energy_level: Energy level (one of: low, medium, high)
- pace_category: Speaking pace (one of: slow, normal, fast)
- emphasis_words: List of words that are emphasized
- mood: Overall mood/feeling
- notes: Any additional observations about speaking style

Return ONLY valid JSON, no other text."""

        # Process with Qwen2-Audio
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        audios = [librosa.load(audio_path, sr=16000)[0]]
        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Parse JSON response
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found")
        except Exception as e:
            print(f"Failed to parse Qwen2-Audio response: {e}")
            print(f"Response was: {response}")
            data = {}
        
        return SemanticFeatures(
            emotion=data.get("emotion", "neutral"),
            emotion_confidence=float(data.get("emotion_confidence", 0.5)),
            tone=data.get("tone", "conversational"),
            energy_level=data.get("energy_level", "medium"),
            pace_category=data.get("pace_category", "normal"),
            emphasis_words=data.get("emphasis_words", []),
            mood=data.get("mood", "neutral"),
            notes=data.get("notes", ""),
        )


class ContourExtractor:
    """
    Extracts pitch contour (F0 trajectory over time).
    This is the "shape" of the intonation.
    """
    
    def __init__(self, min_pitch: float = 75, max_pitch: float = 500):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
    
    def extract(self, audio_path: str) -> PitchContour:
        """Extract pitch contour from audio."""
        
        sound = parselmouth.Sound(audio_path)
        pitch = call(sound, "To Pitch", 0.0, self.min_pitch, self.max_pitch)
        
        # Get time-frequency pairs
        times = []
        values = []
        
        for i in range(pitch.n_frames):
            t = pitch.get_time_from_frame_number(i + 1)
            f0 = pitch.get_value_in_frame(i + 1)
            
            times.append(float(t))
            values.append(float(f0) if not np.isnan(f0) else 0.0)
        
        # Smooth the contour
        values_array = np.array(values)
        values_array[values_array == 0] = np.nan
        
        # Interpolate gaps
        valid_mask = ~np.isnan(values_array)
        if np.any(valid_mask):
            from scipy import interpolate
            valid_indices = np.where(valid_mask)[0]
            valid_values = values_array[valid_mask]
            
            if len(valid_indices) > 1:
                f = interpolate.interp1d(valid_indices, valid_values, 
                                         kind='linear', fill_value='extrapolate')
                smoothed = f(np.arange(len(values_array)))
            else:
                smoothed = values_array.copy()
                smoothed[np.isnan(smoothed)] = valid_values[0] if len(valid_values) > 0 else 0
        else:
            smoothed = np.zeros_like(values_array)
        
        # Apply light smoothing
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(smoothed, sigma=2)
        
        return PitchContour(
            times=times,
            values=values,
            smoothed=smoothed.tolist(),
        )


class CompleteProsodyAnalyzer:
    """
    Complete prosody analyzer that combines all layers:
    - Acoustic (Parselmouth)
    - Rhythm (librosa)
    - Semantic (Qwen2-Audio)
    - Contour (pitch trajectory)
    """
    
    def __init__(self, use_qwen: bool = True, device: str = "cpu"):
        self.acoustic_analyzer = AcousticAnalyzer()
        self.rhythm_analyzer = RhythmAnalyzer()
        self.contour_extractor = ContourExtractor()
        
        self.use_qwen = use_qwen
        if use_qwen:
            self.semantic_analyzer = SemanticAnalyzer(device=device)
        else:
            self.semantic_analyzer = None
    
    def analyze(self, audio_path: str, transcript: str = "") -> ProsodyResult:
        """Run complete prosody analysis."""
        
        print(f"Analyzing: {audio_path}")
        
        # Acoustic features
        print("  Extracting acoustic features...")
        acoustic = self.acoustic_analyzer.analyze(audio_path)
        
        # Rhythm features
        print("  Extracting rhythm features...")
        rhythm = self.rhythm_analyzer.analyze(audio_path)
        
        # Pitch contour
        print("  Extracting pitch contour...")
        contour = self.contour_extractor.extract(audio_path)
        
        # Semantic features (optional, requires Qwen2-Audio)
        semantic = None
        if self.use_qwen and self.semantic_analyzer:
            print("  Analyzing semantic content...")
            try:
                semantic = self.semantic_analyzer.analyze(audio_path, transcript)
            except Exception as e:
                print(f"  Semantic analysis failed: {e}")
        
        return ProsodyResult(
            acoustic=acoustic,
            rhythm=rhythm,
            semantic=semantic,
            contour=contour,
        )


# ============== CLI Interface ==============

def main():
    """Command-line interface for prosody analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze prosody of audio files")
    parser.add_argument("--input", "-i", required=True, help="Input audio file or directory")
    parser.add_argument("--output", "-o", help="Output directory for labels")
    parser.add_argument("--no-qwen", action="store_true", help="Skip Qwen2-Audio analysis")
    parser.add_argument("--device", default="auto", help="Device (cpu, cuda, mps, auto)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create analyzer
    analyzer = CompleteProsodyAnalyzer(use_qwen=not args.no_qwen, device=device)
    
    # Process input
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / "labeled"
    output_path.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("*.wav")) + list(input_path.glob("*.mp3"))
    
    print(f"Found {len(files)} audio files")
    
    for audio_file in files:
        print(f"\nProcessing: {audio_file.name}")
        
        # Analyze
        result = analyzer.analyze(str(audio_file), transcript="")
        
        # Save
        output_file = output_path / f"{audio_file.stem}.json"
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"  Saved: {output_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
