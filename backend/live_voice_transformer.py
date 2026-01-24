"""
Live Voice Transformer

Real-time voice conversion using Seed-VC for emotion transformation.
Speaks normally → Sounds emotional (happy, sad, angry, etc.)
"""

import sys
import time
import threading
import queue
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio

# Add seed-vc to path
SEED_VC_PATH = Path(__file__).parent.parent / "seed-vc"
if SEED_VC_PATH.exists():
    sys.path.insert(0, str(SEED_VC_PATH))


@dataclass
class TransformConfig:
    """Configuration for voice transformation."""
    diffusion_steps: int = 6  # 4-10 for real-time, 25+ for quality
    inference_cfg_rate: float = 0.7  # Style strength (0.0-1.0)
    f0_condition: bool = False  # Use F0 conditioning
    auto_f0_adjust: bool = True  # Auto-adjust pitch
    pitch_shift: int = 0  # Semitones to shift


class LiveVoiceTransformer:
    """
    Real-time voice transformation using Seed-VC.

    Converts voice emotion while preserving identity.
    Uses pre-recorded emotion samples as references.
    """

    def __init__(
        self,
        voice_samples_dir: Optional[Path] = None,
        device: str = "auto",
        preload_emotions: bool = True,
    ):
        """
        Initialize the transformer.

        Args:
            voice_samples_dir: Directory containing emotion reference samples
            device: Device to use ("cuda", "cpu", or "auto")
            preload_emotions: Whether to preload emotion references on init
        """
        self.voice_samples_dir = voice_samples_dir or self._find_voice_samples()
        self.device = self._setup_device(device)

        self.wrapper = None
        self.emotion_samples: Dict[str, Path] = {}
        self.current_emotion: str = "neutral"
        self.current_intensity: float = 0.7
        self.config = TransformConfig()

        # Processing queue for async conversion
        self._input_queue: queue.Queue = queue.Queue()
        self._output_queue: queue.Queue = queue.Queue()
        self._processing_thread: Optional[threading.Thread] = None
        self._running = False

        # Stats
        self._last_inference_time: float = 0
        self._total_processed: int = 0

        # Load model and samples
        self._load_model()
        if preload_emotions:
            self._load_emotion_samples()

    def _setup_device(self, device: str) -> torch.device:
        """Set up the compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        # Handle both string and torch.device input
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    def _find_voice_samples(self) -> Optional[Path]:
        """Find voice samples directory."""
        possible_paths = [
            Path(__file__).parent.parent / "data" / "voice_samples",
            Path.home() / "dev" / "voice-clone-pipeline" / "data" / "voice_samples",
        ]
        for path in possible_paths:
            if path.exists():
                return path
        return None

    def _load_model(self):
        """Load the Seed-VC model."""
        try:
            from seed_vc_wrapper import SeedVCWrapper
            print("Loading Seed-VC model...")
            self.wrapper = SeedVCWrapper(device=self.device)
            print(f"Model loaded on {self.device}")
            print(f"Sample rate: {self.wrapper.sr}")
        except ImportError as e:
            print(f"Error loading Seed-VC: {e}")
            print("Make sure seed-vc is cloned in the project root")
            self.wrapper = None
        except Exception as e:
            print(f"Error initializing Seed-VC: {e}")
            self.wrapper = None

    def _load_emotion_samples(self):
        """Load emotion reference samples from voice_samples directory."""
        if not self.voice_samples_dir or not self.voice_samples_dir.exists():
            print("Voice samples directory not found")
            return

        # Find session directories
        session_dirs = [d for d in self.voice_samples_dir.iterdir() if d.is_dir()]
        if not session_dirs:
            print("No voice sample sessions found")
            return

        # Use most recent session
        session_dir = sorted(session_dirs, reverse=True)[0]
        print(f"Using voice samples from: {session_dir.name}")

        # Find samples for each emotion
        for wav_file in session_dir.glob("*.wav"):
            if "_raw" in wav_file.name:
                continue

            emotion = wav_file.stem.split("_")[0].lower()
            if emotion not in self.emotion_samples:
                self.emotion_samples[emotion] = wav_file

        print(f"Loaded {len(self.emotion_samples)} emotion references:")
        for emotion, path in sorted(self.emotion_samples.items()):
            print(f"  {emotion}: {path.name}")

    @property
    def available_emotions(self) -> list:
        """Get list of available emotions."""
        return sorted(self.emotion_samples.keys())

    @property
    def is_ready(self) -> bool:
        """Check if transformer is ready."""
        return self.wrapper is not None and len(self.emotion_samples) > 0

    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        return self.wrapper.sr if self.wrapper else 22050

    def set_emotion(self, emotion: str, intensity: float = 0.7):
        """
        Set the target emotion for transformation.

        Args:
            emotion: Target emotion (happy, sad, angry, calm, etc.)
            intensity: Emotion strength (0.0-1.0)
        """
        emotion = emotion.lower()
        if emotion not in self.emotion_samples and emotion != "neutral":
            available = ", ".join(self.available_emotions)
            raise ValueError(f"Unknown emotion: {emotion}. Available: {available}")

        self.current_emotion = emotion
        self.current_intensity = max(0.0, min(1.0, intensity))

        # Adjust config based on intensity
        self.config.inference_cfg_rate = 0.5 + (self.current_intensity * 0.4)

    def set_diffusion_steps(self, steps: int):
        """
        Set diffusion steps (quality vs speed tradeoff).

        Args:
            steps: Number of diffusion steps (4=fast, 6=balanced, 10=quality)
        """
        self.config.diffusion_steps = max(2, min(25, steps))

    def convert_audio(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        target_emotion: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert audio to target emotion.

        Args:
            source_audio: Input audio as numpy array
            source_sr: Source sample rate
            target_emotion: Target emotion (uses current if None)

        Returns:
            Tuple of (converted_audio, sample_rate)
        """
        if not self.is_ready:
            raise RuntimeError("Transformer not ready. Check model and samples.")

        emotion = target_emotion or self.current_emotion

        # If neutral or no reference, return original
        if emotion == "neutral" or emotion not in self.emotion_samples:
            return source_audio, source_sr

        # Save source to temp file (Seed-VC expects file paths)
        import tempfile
        import scipy.io.wavfile as wavfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            source_path = f.name
            # Ensure int16 for wav
            if source_audio.dtype != np.int16:
                if source_audio.dtype == np.float32 or source_audio.dtype == np.float64:
                    source_audio = (source_audio * 32767).astype(np.int16)
            wavfile.write(source_path, source_sr, source_audio)

        try:
            # Get reference sample
            reference_path = str(self.emotion_samples[emotion])

            # Perform conversion
            start_time = time.time()

            result = None
            for chunk in self.wrapper.convert_voice(
                source=source_path,
                target=reference_path,
                diffusion_steps=self.config.diffusion_steps,
                inference_cfg_rate=self.config.inference_cfg_rate,
                f0_condition=self.config.f0_condition,
                auto_f0_adjust=self.config.auto_f0_adjust,
                pitch_shift=self.config.pitch_shift,
                stream_output=True,
            ):
                mp3_bytes, audio_tuple = chunk
                result = audio_tuple

            self._last_inference_time = time.time() - start_time
            self._total_processed += 1

            if result is not None:
                # Result is (sample_rate, audio_data)
                output_sr, output_audio = result
                return output_audio, output_sr
            else:
                return source_audio, source_sr

        finally:
            # Clean up temp file
            import os
            os.unlink(source_path)

    def convert_file(
        self,
        source_path: str,
        output_path: str,
        target_emotion: Optional[str] = None,
    ) -> float:
        """
        Convert an audio file to target emotion.

        Args:
            source_path: Path to source audio
            output_path: Path to save output
            target_emotion: Target emotion (uses current if None)

        Returns:
            Inference time in seconds
        """
        import scipy.io.wavfile as wavfile

        # Load source
        sr, audio = wavfile.read(source_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32767.0

        # Convert
        converted, output_sr = self.convert_audio(audio, sr, target_emotion)

        # Save output
        if converted.dtype == np.float32 or converted.dtype == np.float64:
            converted = (converted * 32767).astype(np.int16)
        wavfile.write(output_path, output_sr, converted)

        return self._last_inference_time

    def get_stats(self) -> dict:
        """Get transformation statistics."""
        return {
            "is_ready": self.is_ready,
            "device": str(self.device),
            "current_emotion": self.current_emotion,
            "current_intensity": self.current_intensity,
            "available_emotions": self.available_emotions,
            "sample_rate": self.sample_rate,
            "last_inference_time": self._last_inference_time,
            "total_processed": self._total_processed,
            "config": {
                "diffusion_steps": self.config.diffusion_steps,
                "inference_cfg_rate": self.config.inference_cfg_rate,
            }
        }

    # ===== Async Processing for Real-time =====

    def start_processing(self):
        """Start the async processing thread."""
        if self._running:
            return

        self._running = True
        self._processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processing_thread.start()
        print("Live voice transformer started")

    def stop_processing(self):
        """Stop the async processing thread."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
            self._processing_thread = None
        print("Live voice transformer stopped")

    def _process_loop(self):
        """Background processing loop."""
        while self._running:
            try:
                # Get input from queue (with timeout to allow checking _running)
                item = self._input_queue.get(timeout=0.1)
                if item is None:
                    continue

                request_id, audio, sr, emotion = item

                # Process
                try:
                    converted, output_sr = self.convert_audio(audio, sr, emotion)
                    self._output_queue.put((request_id, converted, output_sr, None))
                except Exception as e:
                    self._output_queue.put((request_id, None, None, str(e)))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def submit_audio(
        self,
        request_id: str,
        audio: np.ndarray,
        sample_rate: int,
        emotion: Optional[str] = None,
    ):
        """
        Submit audio for async processing.

        Args:
            request_id: Unique ID for this request
            audio: Input audio data
            sample_rate: Input sample rate
            emotion: Target emotion (uses current if None)
        """
        self._input_queue.put((request_id, audio, sample_rate, emotion))

    def get_result(self, timeout: float = 0.1) -> Optional[Tuple]:
        """
        Get a processed result.

        Returns:
            Tuple of (request_id, audio, sample_rate, error) or None
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# Singleton instance
_transformer: Optional[LiveVoiceTransformer] = None


def get_transformer() -> LiveVoiceTransformer:
    """Get or create the global transformer instance."""
    global _transformer
    if _transformer is None:
        _transformer = LiveVoiceTransformer()
    return _transformer


# ===== Test =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live Voice Transformer")
    parser.add_argument("--test", action="store_true", help="Run test")
    parser.add_argument("--source", type=str, help="Source audio file")
    parser.add_argument("--emotion", type=str, default="happy", help="Target emotion")
    parser.add_argument("--output", type=str, help="Output file")
    args = parser.parse_args()

    transformer = LiveVoiceTransformer()

    print("\n" + "=" * 60)
    print("LIVE VOICE TRANSFORMER")
    print("=" * 60)
    print(f"Ready: {transformer.is_ready}")
    print(f"Device: {transformer.device}")
    print(f"Emotions: {transformer.available_emotions}")

    if args.test or args.source:
        if not transformer.is_ready:
            print("Transformer not ready!")
            sys.exit(1)

        # Use first available sample if no source specified
        if not args.source:
            samples = list(transformer.emotion_samples.values())
            if samples:
                args.source = str(samples[0])
                print(f"Using sample: {args.source}")

        if args.source:
            output = args.output or f"transformed_{args.emotion}.wav"
            print(f"\nConverting to {args.emotion}...")
            inference_time = transformer.convert_file(args.source, output, args.emotion)
            print(f"Done in {inference_time:.2f}s")
            print(f"Output: {output}")
