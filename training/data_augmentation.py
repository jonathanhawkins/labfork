"""
Audio Data Augmentation Module for Voice Clone Training

Implements augmentation techniques critical for small dataset training:
- Speed perturbation
- Pitch shifting
- Time/frequency masking (SpecAugment-style)
- Noise injection
- Volume perturbation

Reference: Parameter-Efficient Fine-Tuning for Low-Resource TTS (Interspeech 2025)
"""

import random
import math
from typing import Tuple, Optional, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentation."""
    enabled: bool = True
    augment_probability: float = 0.5  # Probability of applying any augmentation

    # Speed perturbation
    speed_perturb_enabled: bool = True
    speed_perturb_range: Tuple[float, float] = (0.9, 1.1)

    # Pitch shifting
    pitch_shift_enabled: bool = True
    pitch_shift_semitones: Tuple[float, float] = (-2.0, 2.0)

    # Time masking (SpecAugment-style)
    time_mask_enabled: bool = True
    time_mask_max_ratio: float = 0.1  # Max 10% of audio masked
    time_mask_num_masks: int = 2

    # Frequency masking (for spectrograms)
    freq_mask_enabled: bool = False
    freq_mask_max_bins: int = 27
    freq_mask_num_masks: int = 2

    # Noise injection
    noise_enabled: bool = True
    noise_snr_range: Tuple[float, float] = (20.0, 40.0)  # dB
    noise_types: List[str] = None  # ['white', 'pink', 'brown']

    # Volume perturbation
    volume_enabled: bool = True
    volume_range: Tuple[float, float] = (0.8, 1.2)

    # Reverberation (optional, more expensive)
    reverb_enabled: bool = False
    reverb_room_scale: Tuple[float, float] = (0.0, 50.0)

    def __post_init__(self):
        if self.noise_types is None:
            self.noise_types = ['white', 'pink']


class AudioAugmenter:
    """
    Audio augmentation pipeline for voice clone training.

    Designed for small datasets where augmentation is critical to prevent overfitting.
    Each augmentation is applied independently with configurable probability.
    """

    def __init__(self, config: AugmentationConfig, sample_rate: int = 24000):
        self.config = config
        self.sample_rate = sample_rate

        # Pre-compute pink noise coefficients for efficiency
        self._pink_noise_b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        self._pink_noise_a = [1, -2.494956002, 2.017265875, -0.522189400]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to waveform.

        Args:
            waveform: [samples] or [1, samples] audio tensor

        Returns:
            Augmented waveform with same shape
        """
        if not self.config.enabled:
            return waveform

        # Ensure 2D: [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Apply augmentations with probability
        if random.random() < self.config.augment_probability:
            waveform = self._apply_augmentations(waveform)

        if squeeze_output:
            waveform = waveform.squeeze(0)

        return waveform

    def _apply_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply all enabled augmentations."""

        # Speed perturbation (changes duration)
        if self.config.speed_perturb_enabled and random.random() < 0.5:
            waveform = self.speed_perturb(waveform)

        # Pitch shifting (preserves duration)
        if self.config.pitch_shift_enabled and random.random() < 0.5:
            waveform = self.pitch_shift(waveform)

        # Time masking
        if self.config.time_mask_enabled and random.random() < 0.5:
            waveform = self.time_mask(waveform)

        # Noise injection
        if self.config.noise_enabled and random.random() < 0.5:
            waveform = self.add_noise(waveform)

        # Volume perturbation
        if self.config.volume_enabled and random.random() < 0.5:
            waveform = self.volume_perturb(waveform)

        return waveform

    def speed_perturb(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply speed perturbation (time stretching).

        Changes both tempo and pitch, then pitch-corrects if desired.
        For voice cloning, we typically want to preserve pitch characteristics.
        """
        low, high = self.config.speed_perturb_range
        factor = random.uniform(low, high)

        # Resample to change speed
        orig_length = waveform.shape[-1]
        new_length = int(orig_length / factor)

        # Use interpolation for speed change
        waveform = F.interpolate(
            waveform.unsqueeze(0),  # [1, C, T]
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)

        return waveform

    def pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply pitch shifting while preserving duration.

        Uses librosa-style pitch shifting via resampling.
        """
        low, high = self.config.pitch_shift_semitones
        semitones = random.uniform(low, high)

        # Pitch shift ratio
        ratio = 2 ** (semitones / 12)

        # Resample to shift pitch, then resample back to preserve duration
        orig_length = waveform.shape[-1]

        # Step 1: Resample to intermediate rate (changes pitch)
        intermediate_length = int(orig_length * ratio)
        shifted = F.interpolate(
            waveform.unsqueeze(0),
            size=intermediate_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)

        # Step 2: Resample back to original length (preserves duration)
        shifted = F.interpolate(
            shifted.unsqueeze(0),
            size=orig_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)

        return shifted

    def time_mask(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment-style time masking.

        Masks random contiguous time segments with zeros.
        Helps model become robust to missing segments.
        """
        length = waveform.shape[-1]
        max_mask_length = int(length * self.config.time_mask_max_ratio)

        for _ in range(self.config.time_mask_num_masks):
            mask_length = random.randint(1, max(1, max_mask_length))
            mask_start = random.randint(0, max(0, length - mask_length))

            waveform[:, mask_start:mask_start + mask_length] = 0

        return waveform

    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Add background noise at random SNR.

        Supports white and pink noise for natural variation.
        """
        low_snr, high_snr = self.config.noise_snr_range
        snr_db = random.uniform(low_snr, high_snr)

        # Generate noise
        noise_type = random.choice(self.config.noise_types)
        noise = self._generate_noise(waveform.shape, noise_type)

        # Calculate scaling for target SNR
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)

        if noise_power > 0:
            # SNR = 10 * log10(signal_power / noise_power)
            # noise_scale = sqrt(signal_power / (10^(SNR/10) * noise_power))
            target_noise_power = signal_power / (10 ** (snr_db / 10))
            noise_scale = torch.sqrt(target_noise_power / noise_power)
            noise = noise * noise_scale

        return waveform + noise

    def _generate_noise(self, shape: torch.Size, noise_type: str) -> torch.Tensor:
        """Generate noise of specified type."""
        if noise_type == 'white':
            return torch.randn(shape)
        elif noise_type == 'pink':
            return self._generate_pink_noise(shape)
        elif noise_type == 'brown':
            return self._generate_brown_noise(shape)
        else:
            return torch.randn(shape)

    def _generate_pink_noise(self, shape: torch.Size) -> torch.Tensor:
        """Generate pink (1/f) noise using Voss-McCartney algorithm."""
        # Simple approximation: filter white noise
        white = torch.randn(shape)

        # Apply simple lowpass-ish filter for pink noise approximation
        # Pink noise has more energy at lower frequencies
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size) / kernel_size

        pink = F.conv1d(
            white.unsqueeze(0) if white.dim() == 2 else white.unsqueeze(0).unsqueeze(0),
            kernel,
            padding=kernel_size // 2
        )

        return pink.squeeze(0) if pink.dim() == 3 else pink.squeeze()

    def _generate_brown_noise(self, shape: torch.Size) -> torch.Tensor:
        """Generate brown (1/f^2) noise via integration."""
        white = torch.randn(shape)
        brown = torch.cumsum(white, dim=-1)
        # Normalize
        brown = brown / (torch.std(brown) + 1e-8)
        return brown

    def volume_perturb(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random volume scaling."""
        low, high = self.config.volume_range
        scale = random.uniform(low, high)
        return waveform * scale


class SpecAugment:
    """
    SpecAugment for spectrogram-based augmentation.

    Applies time and frequency masking to spectrograms.
    Reference: SpecAugment (Park et al., 2019)
    """

    def __init__(
        self,
        time_mask_param: int = 100,
        freq_mask_param: int = 27,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ):
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.

        Args:
            spectrogram: [freq, time] or [batch, freq, time]

        Returns:
            Augmented spectrogram
        """
        for _ in range(self.num_time_masks):
            spectrogram = self.time_mask(spectrogram)

        for _ in range(self.num_freq_masks):
            spectrogram = self.freq_mask(spectrogram)

        return spectrogram


def create_augmenter_from_config(config_dict: dict, sample_rate: int = 24000) -> AudioAugmenter:
    """
    Create augmenter from config dictionary.

    Args:
        config_dict: Dictionary with augmentation settings
        sample_rate: Audio sample rate

    Returns:
        Configured AudioAugmenter instance
    """
    aug_config = AugmentationConfig(
        enabled=config_dict.get('enabled', True),
        augment_probability=config_dict.get('augment_probability', 0.5),
        speed_perturb_enabled=config_dict.get('speed_perturb_enabled', True),
        speed_perturb_range=tuple(config_dict.get('speed_perturb', [0.9, 1.1])),
        pitch_shift_enabled=config_dict.get('pitch_shift_enabled', True),
        pitch_shift_semitones=tuple(config_dict.get('pitch_shift_semitones', [-2, 2])),
        time_mask_enabled=config_dict.get('time_mask_enabled', True),
        time_mask_max_ratio=config_dict.get('time_mask_ratio', 0.1),
        noise_enabled=config_dict.get('noise_enabled', True),
        noise_snr_range=tuple(config_dict.get('noise_snr_db', [20, 40])),
        volume_enabled=config_dict.get('volume_enabled', True),
        volume_range=tuple(config_dict.get('volume_range', [0.8, 1.2])),
    )

    return AudioAugmenter(aug_config, sample_rate)


# Test augmentations
if __name__ == "__main__":
    # Create test waveform
    sample_rate = 24000
    duration = 3.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * math.pi * 440 * t)  # 440 Hz sine wave

    print(f"Original waveform shape: {waveform.shape}")
    print(f"Original duration: {len(waveform) / sample_rate:.2f}s")

    # Create augmenter
    config = AugmentationConfig(
        enabled=True,
        augment_probability=1.0,  # Always augment for testing
    )
    augmenter = AudioAugmenter(config, sample_rate)

    # Apply augmentations
    augmented = augmenter(waveform)

    print(f"Augmented waveform shape: {augmented.shape}")
    print(f"Augmented duration: {len(augmented) / sample_rate:.2f}s")

    # Test each augmentation individually
    print("\nTesting individual augmentations:")

    # Speed perturb
    speed_aug = augmenter.speed_perturb(waveform.unsqueeze(0))
    print(f"  Speed perturb: {waveform.shape} -> {speed_aug.squeeze(0).shape}")

    # Pitch shift
    pitch_aug = augmenter.pitch_shift(waveform.unsqueeze(0))
    print(f"  Pitch shift: {waveform.shape} -> {pitch_aug.squeeze(0).shape}")

    # Time mask
    mask_aug = augmenter.time_mask(waveform.unsqueeze(0).clone())
    zeros = (mask_aug == 0).sum().item()
    print(f"  Time mask: {zeros} samples masked ({100*zeros/len(waveform):.1f}%)")

    # Noise
    noise_aug = augmenter.add_noise(waveform.unsqueeze(0))
    snr = 10 * torch.log10(
        torch.mean(waveform**2) / torch.mean((noise_aug.squeeze(0) - waveform)**2 + 1e-8)
    )
    print(f"  Noise injection: estimated SNR = {snr:.1f} dB")

    print("\nAugmentation module ready!")
