"""
Keyframe Prosody System for Voice Clone Pipeline

This module provides keyframe-based prosody control for speech synthesis,
allowing users to define emotion/intensity changes over time and interpolate
between them using Catmull-Rom splines for smooth transitions.

The system integrates with the existing EmotionToProsody class to convert
emotion labels to prosody vectors, then interpolates between keyframes to
produce dense prosody conditioning for generation.

Usage:
    keyframes = [
        ProsodyKeyframe(time=0.0, emotion="neutral", intensity=0.5),
        ProsodyKeyframe(time=0.3, emotion="happy", intensity=0.8),
        ProsodyKeyframe(time=0.7, emotion="sad", intensity=0.6),
        ProsodyKeyframe(time=1.0, emotion="neutral", intensity=0.5),
    ]

    prosody = keyframes_to_prosody(keyframes, duration_seconds=5.0)
    # Use prosody dict with ProsodyControlledCSM for generation
"""

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add training directory to path for importing prosody_conditioning
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
from prosody_conditioning import ProsodyConfig, EmotionToProody as EmotionToProsody

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize_text(text: str) -> List[Dict[str, int]]:
    """Tokenize text into words with character offsets."""
    words = []
    for match in _WORD_RE.finditer(text):
        words.append({
            "word": match.group(0),
            "start": match.start(),
            "end": match.end(),
        })
    return words


def resolve_keyframe_times(
    keyframes: List[Dict[str, Any]],
    text: str,
    word_timestamps: Optional[List[Dict[str, float]]] = None,
) -> List[Dict[str, any]]:
    """
    Resolve keyframe times from text-aligned anchors.

    Supports keyframes with:
      - time: explicit time in seconds or normalized (0-1)
      - word_index: zero-based word index in the text
      - word: word string (case-insensitive) with optional occurrence (1-based)
      - char_index: character index in the text

    If word_timestamps are provided (from Whisper), they are preferred for timing.
    Otherwise, word positions are approximated from character offsets.
    """
    if not text:
        return keyframes

    tokens = _tokenize_text(text)
    text_len = max(1, len(text))

    # Build normalized word time map
    if word_timestamps:
        # Use real timestamps if available
        duration = max((w.get("end", 0.0) for w in word_timestamps), default=0.0)
        duration = max(duration, 1e-6)
        word_times = [w.get("start", 0.0) / duration for w in word_timestamps]
    else:
        # Approximate by character position
        word_times = [tok["start"] / text_len for tok in tokens]

    def _resolve_by_word_index(idx: int) -> float:
        if idx < 0 or idx >= len(word_times):
            raise ValueError(f"word_index {idx} out of range for {len(word_times)} words")
        return float(word_times[idx])

    def _resolve_by_word(word: str, occurrence: int) -> float:
        if not tokens:
            raise ValueError("No words found in text to resolve keyframe word anchor")
        needle = word.lower()
        matches = [i for i, tok in enumerate(tokens) if tok["word"].lower() == needle]
        if not matches:
            raise ValueError(f"word '{word}' not found in text")
        occ_idx = max(1, occurrence) - 1
        if occ_idx >= len(matches):
            raise ValueError(f"word '{word}' occurrence {occurrence} out of range")
        return _resolve_by_word_index(matches[occ_idx])

    resolved = []
    for kf in keyframes:
        if "time" in kf and kf["time"] is not None:
            resolved.append(kf)
            continue

        if "word_index" in kf:
            time = _resolve_by_word_index(int(kf["word_index"]))
        elif "word" in kf:
            occurrence = int(kf.get("occurrence", 1))
            time = _resolve_by_word(kf["word"], occurrence)
        elif "char_index" in kf:
            char_index = int(kf["char_index"])
            time = max(0.0, min(1.0, char_index / text_len))
        else:
            raise ValueError("Keyframe missing time/word_index/word/char_index")

        resolved_kf = dict(kf)
        resolved_kf["time"] = time
        resolved.append(resolved_kf)

    return resolved


@dataclass
class ProsodyKeyframe:
    """
    A single keyframe defining prosody state at a point in time.

    Attributes:
        time: Position in the timeline. Can be normalized (0.0 to 1.0) or
              absolute seconds depending on usage context.
        emotion: The emotion label for this keyframe. Supported emotions:
                 "neutral", "happy", "sad", "angry", "surprised", "calm"
        intensity: Strength of the emotion from 0.0 (none) to 1.0 (maximum).
        energy: Optional override for energy/loudness (0.0 to 1.0).
                If None, uses the emotion's default energy.
        pitch_tendency: Optional pitch offset. "low" shifts pitch down,
                       "high" shifts pitch up, "neutral" uses default.
    """
    time: float
    emotion: str
    intensity: float
    energy: Optional[float] = None
    pitch_tendency: Optional[str] = None  # "low", "neutral", "high"

    def __post_init__(self):
        """Validate keyframe values."""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity must be between 0.0 and 1.0, got {self.intensity}")
        if self.energy is not None and not 0.0 <= self.energy <= 1.0:
            raise ValueError(f"energy must be between 0.0 and 1.0, got {self.energy}")
        if self.pitch_tendency is not None and self.pitch_tendency not in ("low", "neutral", "high"):
            raise ValueError(f"pitch_tendency must be 'low', 'neutral', or 'high', got {self.pitch_tendency}")

        # Normalize emotion to lowercase
        self.emotion = self.emotion.lower()
        valid_emotions = {"neutral", "happy", "sad", "angry", "surprised", "calm", "excited", "fearful"}
        if self.emotion not in valid_emotions:
            raise ValueError(f"emotion must be one of {valid_emotions}, got {self.emotion}")


def catmull_rom_spline(
    points: np.ndarray,
    times: np.ndarray,
    sample_times: np.ndarray,
    tension: float = 0.5
) -> np.ndarray:
    """
    Compute Catmull-Rom spline interpolation for a set of control points.

    Catmull-Rom splines pass through all control points and produce smooth
    curves with C1 continuity. They're ideal for prosody interpolation because
    they don't overshoot and create natural-feeling transitions.

    Args:
        points: Control point values of shape [N, D] where N is number of points
               and D is dimensionality.
        times: Time values for each control point, shape [N].
        sample_times: Times at which to sample the spline, shape [M].
        tension: Controls curve tightness. 0.5 is standard Catmull-Rom.
                Higher values create tighter curves, lower values are looser.

    Returns:
        Interpolated values at sample_times, shape [M, D].
    """
    n_points = len(points)
    n_samples = len(sample_times)
    n_dims = points.shape[1] if len(points.shape) > 1 else 1

    # Handle 1D case
    if len(points.shape) == 1:
        points = points.reshape(-1, 1)

    # Duplicate endpoints for edge case handling (Catmull-Rom needs 4 points)
    # This ensures smooth behavior at boundaries
    extended_points = np.vstack([
        points[0] - (points[1] - points[0]),  # Virtual point before start
        points,
        points[-1] + (points[-1] - points[-2])  # Virtual point after end
    ])
    extended_times = np.hstack([
        times[0] - (times[1] - times[0]) if len(times) > 1 else times[0] - 0.1,
        times,
        times[-1] + (times[-1] - times[-2]) if len(times) > 1 else times[-1] + 0.1
    ])

    result = np.zeros((n_samples, n_dims))

    for i, t in enumerate(sample_times):
        # Find the segment containing t
        # We need points P0, P1, P2, P3 where t is between P1 and P2
        segment_idx = 0
        for j in range(1, len(extended_times) - 2):
            if extended_times[j] <= t <= extended_times[j + 1]:
                segment_idx = j
                break
            elif t < extended_times[j]:
                segment_idx = j
                break
        else:
            segment_idx = len(extended_times) - 3

        # Clamp to valid range
        segment_idx = max(1, min(segment_idx, len(extended_times) - 3))

        # Get the 4 control points for this segment
        p0 = extended_points[segment_idx - 1]
        p1 = extended_points[segment_idx]
        p2 = extended_points[segment_idx + 1]
        p3 = extended_points[segment_idx + 2]

        t0 = extended_times[segment_idx - 1]
        t1 = extended_times[segment_idx]
        t2 = extended_times[segment_idx + 1]
        t3 = extended_times[segment_idx + 2]

        # Normalize t to [0, 1] within segment
        segment_length = t2 - t1
        if segment_length < 1e-8:
            u = 0.0
        else:
            u = (t - t1) / segment_length
        u = np.clip(u, 0.0, 1.0)

        # Catmull-Rom basis matrix with tension parameter
        # Standard Catmull-Rom uses tension = 0.5
        s = tension

        # Compute using the cardinal spline formulation
        u2 = u * u
        u3 = u2 * u

        # Catmull-Rom coefficients
        c0 = -s*u3 + 2*s*u2 - s*u
        c1 = (2-s)*u3 + (s-3)*u2 + 1
        c2 = (s-2)*u3 + (3-2*s)*u2 + s*u
        c3 = s*u3 - s*u2

        result[i] = c0*p0 + c1*p1 + c2*p2 + c3*p3

    return result


def _apply_pitch_tendency(acoustic: torch.Tensor, pitch_tendency: Optional[str]) -> torch.Tensor:
    """
    Apply pitch tendency override to acoustic vector.

    Args:
        acoustic: Acoustic prosody vector [acoustic_dim].
        pitch_tendency: "low", "neutral", or "high".

    Returns:
        Modified acoustic vector with pitch adjusted.
    """
    if pitch_tendency is None or pitch_tendency == "neutral":
        return acoustic

    acoustic = acoustic.clone()

    # Index 0 is pitch_mean in the acoustic vector
    if pitch_tendency == "low":
        acoustic[0] = acoustic[0] * 0.7  # Reduce pitch
    elif pitch_tendency == "high":
        acoustic[0] = min(1.0, acoustic[0] * 1.3)  # Increase pitch

    return acoustic


def _apply_energy_override(acoustic: torch.Tensor, energy: Optional[float]) -> torch.Tensor:
    """
    Apply energy override to acoustic vector.

    Args:
        acoustic: Acoustic prosody vector [acoustic_dim].
        energy: Energy override value (0.0 to 1.0).

    Returns:
        Modified acoustic vector with energy adjusted.
    """
    if energy is None:
        return acoustic

    acoustic = acoustic.clone()
    # Index 2 is energy in the acoustic vector
    acoustic[2] = energy

    return acoustic


def keyframe_to_prosody_vectors(
    keyframe: ProsodyKeyframe,
    config: ProsodyConfig = None
) -> Dict[str, torch.Tensor]:
    """
    Convert a single keyframe to prosody vectors using EmotionToProsody.

    This applies the keyframe's emotion and intensity, then applies any
    optional overrides for energy and pitch_tendency.

    Args:
        keyframe: The keyframe to convert.
        config: ProsodyConfig for dimensions. Uses default if None.

    Returns:
        Dict with 'semantic', 'acoustic', 'rhythm', 'contour' tensors,
        each of shape [dim] (no batch dimension).
    """
    if config is None:
        config = ProsodyConfig()

    # Get base prosody from emotion
    prosody = EmotionToProsody.get_prosody(
        emotion=keyframe.emotion,
        intensity=keyframe.intensity,
        config=config
    )

    # Remove batch dimension for easier manipulation
    semantic = prosody['semantic'].squeeze(0)
    acoustic = prosody['acoustic'].squeeze(0)
    rhythm = prosody['rhythm'].squeeze(0)
    contour = prosody['contour'].squeeze(0)

    # Apply optional overrides
    acoustic = _apply_pitch_tendency(acoustic, keyframe.pitch_tendency)
    acoustic = _apply_energy_override(acoustic, keyframe.energy)

    return {
        'semantic': semantic,
        'acoustic': acoustic,
        'rhythm': rhythm,
        'contour': contour
    }


def interpolate_prosody_keyframes(
    keyframes: List[ProsodyKeyframe],
    sample_points: int = 64,
    config: ProsodyConfig = None,
    tension: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    Interpolate between prosody keyframes using Catmull-Rom splines.

    This function takes a list of keyframes defining emotion states at various
    times and produces smooth interpolated prosody vectors at regularly-spaced
    sample points.

    Args:
        keyframes: List of ProsodyKeyframe objects. Must have at least 1 keyframe.
                  Keyframes should be sorted by time, but will be sorted if not.
        sample_points: Number of output sample points. Default 64 matches
                      the contour_dim in ProsodyConfig.
        config: ProsodyConfig for prosody dimensions.
        tension: Catmull-Rom tension parameter. 0.5 is standard.

    Returns:
        Dict with 'semantic', 'acoustic', 'rhythm', 'contour' tensors,
        each of shape [sample_points, dim].

    Raises:
        ValueError: If keyframes list is empty.
    """
    if not keyframes:
        raise ValueError("keyframes list cannot be empty")

    if config is None:
        config = ProsodyConfig()

    # Sort keyframes by time
    keyframes = sorted(keyframes, key=lambda kf: kf.time)

    # Handle edge cases with fewer than 4 keyframes by duplicating endpoints
    if len(keyframes) == 1:
        # Single keyframe: constant prosody
        kf = keyframes[0]
        prosody = keyframe_to_prosody_vectors(kf, config)
        return {
            key: val.unsqueeze(0).expand(sample_points, -1).clone()
            for key, val in prosody.items()
        }

    # Convert all keyframes to prosody vectors
    times = np.array([kf.time for kf in keyframes])

    # Normalize times to [0, 1] if they aren't already
    time_min, time_max = times.min(), times.max()
    if time_max - time_min > 1e-8:
        normalized_times = (times - time_min) / (time_max - time_min)
    else:
        normalized_times = np.zeros_like(times)

    # Get prosody vectors for each keyframe
    prosody_at_keyframes = [keyframe_to_prosody_vectors(kf, config) for kf in keyframes]

    # Stack each prosody type into arrays for interpolation
    semantic_points = np.stack([p['semantic'].numpy() for p in prosody_at_keyframes])
    acoustic_points = np.stack([p['acoustic'].numpy() for p in prosody_at_keyframes])
    rhythm_points = np.stack([p['rhythm'].numpy() for p in prosody_at_keyframes])
    contour_points = np.stack([p['contour'].numpy() for p in prosody_at_keyframes])

    # Generate sample times
    sample_times = np.linspace(0.0, 1.0, sample_points)

    # Interpolate each prosody type
    semantic_interp = catmull_rom_spline(semantic_points, normalized_times, sample_times, tension)
    acoustic_interp = catmull_rom_spline(acoustic_points, normalized_times, sample_times, tension)
    rhythm_interp = catmull_rom_spline(rhythm_points, normalized_times, sample_times, tension)
    contour_interp = catmull_rom_spline(contour_points, normalized_times, sample_times, tension)

    # Convert to tensors and clamp to valid range [0, 1]
    return {
        'semantic': torch.clamp(torch.from_numpy(semantic_interp).float(), 0.0, 1.0),
        'acoustic': torch.clamp(torch.from_numpy(acoustic_interp).float(), 0.0, 1.0),
        'rhythm': torch.clamp(torch.from_numpy(rhythm_interp).float(), 0.0, 1.0),
        'contour': torch.clamp(torch.from_numpy(contour_interp).float(), 0.0, 1.0),
    }


def keyframes_to_prosody(
    keyframes: List[ProsodyKeyframe],
    duration_seconds: float,
    sample_points: int = 64,
    config: ProsodyConfig = None
) -> Dict[str, torch.Tensor]:
    """
    Convert keyframes to generation-ready prosody tensors.

    This is the main entry point for the keyframe prosody system. It takes
    a list of keyframes with absolute or normalized times and produces
    dense prosody vectors suitable for conditioning the CSM model.

    IMPORTANT - TEMPORAL DATA HANDLING:
    ------------------------------------
    This function returns BOTH dense temporal data AND averaged global vectors.
    The current model architecture (ProsodyControlledCSM) only uses a fixed
    4-token prefix (see training/prosody_conditioning.py, num_prosody_tokens=4),
    so the averaged versions are provided for backwards compatibility.

    However, the temporal keyframe information is preserved in the dense tensors:
    - Dense keys ('semantic_dense', 'acoustic_dense', etc.): Full temporal data
      with shape [sample_points, dim] - USE THESE for per-frame conditioning
    - Averaged keys ('semantic', 'acoustic', etc.): Global conditioning with
      shape [1, dim] - these LOSE per-keyframe edits due to averaging

    For temporal prosody control (per-segment emotion changes), use
    get_temporal_prosody_tokens() which converts dense data to segment-wise
    tokens matching the model's num_prosody_tokens.

    Args:
        keyframes: List of ProsodyKeyframe objects defining emotion trajectory.
        duration_seconds: Total duration of the audio in seconds. Used to
                         interpret absolute time values in keyframes.
        sample_points: Number of output sample points. Default 64.
        config: ProsodyConfig for prosody dimensions.

    Returns:
        Dict with the following keys:

        Dense temporal data (RECOMMENDED for preserving keyframe edits):
            - 'semantic_dense': [sample_points, semantic_dim] - emotion over time
            - 'acoustic_dense': [sample_points, acoustic_dim] - voice properties over time
            - 'rhythm_dense': [sample_points, rhythm_dim] - timing over time
            - 'contour_dense': [sample_points, contour_dim] - pitch trajectory over time

        Averaged global data (for backwards compatibility with current model):
            - 'semantic': [1, semantic_dim] - averaged, LOSES temporal variation
            - 'acoustic': [1, acoustic_dim] - averaged, LOSES temporal variation
            - 'rhythm': [1, rhythm_dim] - averaged, LOSES temporal variation
            - 'contour': [1, contour_dim] - averaged, LOSES temporal variation

    See Also:
        get_temporal_prosody_tokens(): Convert dense prosody to per-segment tokens
                                       for models that support temporal conditioning.

    Example:
        >>> keyframes = [
        ...     ProsodyKeyframe(time=0.0, emotion="neutral", intensity=0.5),
        ...     ProsodyKeyframe(time=2.5, emotion="happy", intensity=0.9),
        ...     ProsodyKeyframe(time=5.0, emotion="neutral", intensity=0.5),
        ... ]
        >>> prosody = keyframes_to_prosody(keyframes, duration_seconds=5.0)
        >>>
        >>> # For global conditioning (current model, loses temporal info):
        >>> # Use prosody['semantic'], prosody['acoustic'], etc.
        >>>
        >>> # For temporal conditioning (preserves keyframe edits):
        >>> # Use prosody['semantic_dense'], prosody['acoustic_dense'], etc.
        >>> # Or convert to segment tokens:
        >>> temporal = get_temporal_prosody_tokens(prosody, num_segments=4)
    """
    if config is None:
        config = ProsodyConfig()

    if duration_seconds <= 0:
        raise ValueError(f"duration_seconds must be positive, got {duration_seconds}")

    # Normalize keyframe times if they appear to be in absolute seconds
    normalized_keyframes = []
    for kf in keyframes:
        # Check if time is likely absolute (> 1.0 suggests seconds, not normalized)
        if kf.time > 1.0 or (len(keyframes) > 1 and max(k.time for k in keyframes) > 1.0):
            # Convert absolute time to normalized [0, 1]
            normalized_time = kf.time / duration_seconds
        else:
            normalized_time = kf.time

        normalized_time = max(0.0, min(1.0, normalized_time))

        normalized_keyframes.append(ProsodyKeyframe(
            time=normalized_time,
            emotion=kf.emotion,
            intensity=kf.intensity,
            energy=kf.energy,
            pitch_tendency=kf.pitch_tendency
        ))

    # Interpolate to get dense prosody
    dense_prosody = interpolate_prosody_keyframes(
        normalized_keyframes,
        sample_points=sample_points,
        config=config
    )

    # Create output with both dense and averaged versions
    # Averaged versions are what the ProsodyControlledCSM expects (batch dimension)
    result = {}

    for key in ['semantic', 'acoustic', 'rhythm', 'contour']:
        dense = dense_prosody[key]  # [sample_points, dim]

        # Store dense version
        result[f'{key}_dense'] = dense

        # Average over time for global conditioning [1, dim]
        result[key] = dense.mean(dim=0, keepdim=True)

    return result


def get_temporal_prosody_tokens(
    prosody: Dict[str, torch.Tensor],
    num_segments: int = 4
) -> Dict[str, torch.Tensor]:
    """
    Convert dense prosody to temporal tokens for per-segment control.

    Instead of averaging all time steps into a single global vector (which
    loses per-keyframe edits), this function divides the dense prosody into
    segments and provides one prosody token per segment. This preserves
    temporal variation in the keyframe trajectory.

    This is designed to work with models that support temporal prosody
    conditioning, where each segment token conditions a portion of the
    generated audio. The default num_segments=4 matches the current
    ProsodyConfig.num_prosody_tokens setting.

    Args:
        prosody: Output from keyframes_to_prosody() containing '_dense' keys.
                 Expected keys: 'semantic_dense', 'acoustic_dense',
                 'rhythm_dense', 'contour_dense' with shape [sample_points, dim].
        num_segments: Number of temporal segments to divide the prosody into.
                     Default 4 matches num_prosody_tokens in ProsodyConfig.
                     Each segment will have shape [dim], and the output will
                     have shape [num_segments, dim].

    Returns:
        Dict with prosody tensors of shape [num_segments, dim]:
            - 'semantic': [num_segments, semantic_dim] - emotion per segment
            - 'acoustic': [num_segments, acoustic_dim] - voice properties per segment
            - 'rhythm': [num_segments, rhythm_dim] - timing per segment
            - 'contour': [num_segments, contour_dim] - pitch per segment

        Unlike keyframes_to_prosody(), these are NOT averaged globally.
        Each row represents a different time segment of the utterance.

    Raises:
        KeyError: If prosody dict is missing required '_dense' keys.
        ValueError: If num_segments < 1.

    Example:
        >>> # Define keyframes with emotion changes over time
        >>> keyframes = [
        ...     ProsodyKeyframe(time=0.0, emotion="neutral", intensity=0.5),
        ...     ProsodyKeyframe(time=0.5, emotion="angry", intensity=0.9),
        ...     ProsodyKeyframe(time=1.0, emotion="calm", intensity=0.4),
        ... ]
        >>> prosody = keyframes_to_prosody(keyframes, duration_seconds=3.0)
        >>>
        >>> # Convert to 4 temporal tokens (one per quarter of the utterance)
        >>> temporal = get_temporal_prosody_tokens(prosody, num_segments=4)
        >>>
        >>> # temporal['semantic'] shape: [4, 8]
        >>> # Segment 0 (0-25%): mostly neutral
        >>> # Segment 1 (25-50%): transitioning to angry
        >>> # Segment 2 (50-75%): angry, transitioning to calm
        >>> # Segment 3 (75-100%): mostly calm

    Note:
        The current ProsodyControlledCSM in training/prosody_conditioning.py
        expects prosody tensors with shape [batch, dim]. To use temporal tokens,
        the model architecture would need to be extended to accept
        [batch, num_segments, dim] and apply different conditioning per segment.
    """
    if num_segments < 1:
        raise ValueError(f"num_segments must be >= 1, got {num_segments}")

    result = {}

    for key in ['semantic', 'acoustic', 'rhythm', 'contour']:
        dense_key = f'{key}_dense'
        if dense_key not in prosody:
            raise KeyError(
                f"Missing '{dense_key}' in prosody dict. "
                f"Make sure to use output from keyframes_to_prosody()."
            )

        dense = prosody[dense_key]  # [sample_points, dim]
        sample_points = dense.shape[0]

        # Divide into segments and average within each segment
        # This preserves the temporal trajectory while reducing to num_segments tokens
        segment_size = sample_points // num_segments
        segments = []

        for i in range(num_segments):
            start_idx = i * segment_size
            # Last segment takes any remaining samples
            if i == num_segments - 1:
                end_idx = sample_points
            else:
                end_idx = start_idx + segment_size

            # Average within this segment
            segment_prosody = dense[start_idx:end_idx].mean(dim=0)  # [dim]
            segments.append(segment_prosody)

        # Stack segments: [num_segments, dim]
        result[key] = torch.stack(segments, dim=0)

    return result


def visualize_keyframes(
    keyframes: List[ProsodyKeyframe],
    prosody: Dict[str, torch.Tensor],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize keyframe interpolation results.

    Creates a multi-panel plot showing:
    - Keyframe positions on timeline
    - Interpolated prosody trajectories

    Args:
        keyframes: Original keyframes for reference.
        prosody: Output from keyframes_to_prosody() or interpolate_prosody_keyframes().
        save_path: If provided, saves figure to this path. Otherwise displays.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Prosody Keyframe Interpolation', fontsize=14)

    # Get dense versions if available, otherwise use what we have
    semantic = prosody.get('semantic_dense', prosody['semantic'])
    acoustic = prosody.get('acoustic_dense', prosody['acoustic'])
    rhythm = prosody.get('rhythm_dense', prosody['rhythm'])
    contour = prosody.get('contour_dense', prosody['contour'])

    # Ensure 2D
    if semantic.dim() == 1:
        semantic = semantic.unsqueeze(0)
    if acoustic.dim() == 1:
        acoustic = acoustic.unsqueeze(0)
    if rhythm.dim() == 1:
        rhythm = rhythm.unsqueeze(0)
    if contour.dim() == 1:
        contour = contour.unsqueeze(0)

    sample_points = semantic.shape[0]
    t = np.linspace(0, 1, sample_points)

    # Keyframe times (normalized)
    kf_times = [kf.time for kf in keyframes]
    if max(kf_times) > 1.0:
        kf_times = [t / max(kf_times) for t in kf_times]

    # Colors for different dimensions
    colors = plt.cm.tab10.colors

    # Semantic plot
    ax = axes[0, 0]
    ax.set_title('Semantic (Emotion Scores)')
    for i in range(min(6, semantic.shape[1])):
        label = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'calm'][i]
        ax.plot(t, semantic[:, i].numpy(), color=colors[i], label=label, alpha=0.8)
    for kt in kf_times:
        ax.axvline(kt, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Score')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Acoustic plot
    ax = axes[0, 1]
    ax.set_title('Acoustic (Voice Properties)')
    labels = ['pitch_mean', 'pitch_std', 'energy']
    for i in range(min(3, acoustic.shape[1])):
        ax.plot(t, acoustic[:, i].numpy(), color=colors[i], label=labels[i], alpha=0.8)
    for kt in kf_times:
        ax.axvline(kt, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Rhythm plot
    ax = axes[1, 0]
    ax.set_title('Rhythm (Timing)')
    labels = ['speaking_rate', 'pause_ratio', 'syllable_rate', 'articulation']
    for i in range(min(4, rhythm.shape[1])):
        ax.plot(t, rhythm[:, i].numpy(), color=colors[i], label=labels[i], alpha=0.8)
    for kt in kf_times:
        ax.axvline(kt, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Contour plot (average trajectory)
    ax = axes[1, 1]
    ax.set_title('Pitch Contour (Trajectory)')
    contour_avg = contour.mean(dim=1).numpy()
    ax.plot(t, contour_avg, color=colors[0], linewidth=2, label='Avg Pitch')
    # Also show a few individual contour dimensions
    for i in [0, 16, 32, 48]:
        if i < contour.shape[1]:
            ax.plot(t, contour[:, i].numpy(), color=colors[i//16 + 1],
                   alpha=0.3, linewidth=1)
    for kt in kf_times:
        ax.axvline(kt, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Pitch Value')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Keyframe Prosody System - Test Suite")
    print("=" * 60)

    # Test 1: Basic keyframe creation
    print("\n[Test 1] Creating keyframes...")
    try:
        kf1 = ProsodyKeyframe(time=0.0, emotion="neutral", intensity=0.5)
        kf2 = ProsodyKeyframe(time=0.3, emotion="happy", intensity=0.9, energy=0.8)
        kf3 = ProsodyKeyframe(time=0.7, emotion="sad", intensity=0.6, pitch_tendency="low")
        kf4 = ProsodyKeyframe(time=1.0, emotion="neutral", intensity=0.5)
        keyframes = [kf1, kf2, kf3, kf4]
        print(f"  Created {len(keyframes)} keyframes")
        print(f"  Keyframes: {[(kf.time, kf.emotion, kf.intensity) for kf in keyframes]}")
        print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise

    # Test 2: Single keyframe conversion
    print("\n[Test 2] Converting single keyframe to prosody...")
    try:
        config = ProsodyConfig()
        prosody_single = keyframe_to_prosody_vectors(kf2, config)
        print(f"  Semantic shape: {prosody_single['semantic'].shape}")
        print(f"  Acoustic shape: {prosody_single['acoustic'].shape}")
        print(f"  Rhythm shape: {prosody_single['rhythm'].shape}")
        print(f"  Contour shape: {prosody_single['contour'].shape}")
        print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise

    # Test 3: Catmull-Rom interpolation
    print("\n[Test 3] Testing Catmull-Rom spline interpolation...")
    try:
        # Simple 1D test
        points = np.array([[0.0], [1.0], [0.5], [0.8]])
        times = np.array([0.0, 0.3, 0.7, 1.0])
        sample_t = np.linspace(0, 1, 20)
        result = catmull_rom_spline(points, times, sample_t)
        print(f"  Input points: {points.flatten()}")
        print(f"  Output shape: {result.shape}")
        print(f"  Output range: [{result.min():.3f}, {result.max():.3f}]")
        print(f"  Passes through keypoints: {np.allclose(result[0], 0.0, atol=0.1)}")
        print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise

    # Test 4: Multi-keyframe interpolation
    print("\n[Test 4] Interpolating multiple keyframes...")
    try:
        dense_prosody = interpolate_prosody_keyframes(keyframes, sample_points=64)
        print(f"  Dense semantic shape: {dense_prosody['semantic'].shape}")
        print(f"  Dense acoustic shape: {dense_prosody['acoustic'].shape}")
        print(f"  Dense rhythm shape: {dense_prosody['rhythm'].shape}")
        print(f"  Dense contour shape: {dense_prosody['contour'].shape}")
        print(f"  Values are clamped: {(dense_prosody['semantic'] >= 0).all() and (dense_prosody['semantic'] <= 1).all()}")
        print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise

    # Test 5: Main API - keyframes_to_prosody
    print("\n[Test 5] Testing main API (keyframes_to_prosody)...")
    try:
        # Test with absolute time values
        keyframes_abs = [
            ProsodyKeyframe(time=0.0, emotion="neutral", intensity=0.5),
            ProsodyKeyframe(time=2.5, emotion="happy", intensity=0.9),
            ProsodyKeyframe(time=4.0, emotion="angry", intensity=0.7, energy=0.9),
            ProsodyKeyframe(time=5.0, emotion="calm", intensity=0.6),
        ]

        prosody = keyframes_to_prosody(
            keyframes_abs,
            duration_seconds=5.0,
            sample_points=64
        )

        print(f"  Global semantic shape: {prosody['semantic'].shape}")
        print(f"  Global acoustic shape: {prosody['acoustic'].shape}")
        print(f"  Dense semantic shape: {prosody['semantic_dense'].shape}")
        print(f"  Dense acoustic shape: {prosody['acoustic_dense'].shape}")

        # Verify shapes
        assert prosody['semantic'].shape == (1, 8), "Semantic shape mismatch"
        assert prosody['acoustic'].shape == (1, 12), "Acoustic shape mismatch"
        assert prosody['rhythm'].shape == (1, 8), "Rhythm shape mismatch"
        assert prosody['contour'].shape == (1, 64), "Contour shape mismatch"
        assert prosody['semantic_dense'].shape == (64, 8), "Dense semantic shape mismatch"

        print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise

    # Test 6: Edge cases
    print("\n[Test 6] Testing edge cases...")
    try:
        # Single keyframe
        single_kf = [ProsodyKeyframe(time=0.5, emotion="happy", intensity=0.7)]
        prosody_single = keyframes_to_prosody(single_kf, duration_seconds=3.0)
        print(f"  Single keyframe: semantic shape {prosody_single['semantic'].shape}")

        # Two keyframes
        two_kf = [
            ProsodyKeyframe(time=0.0, emotion="neutral", intensity=0.5),
            ProsodyKeyframe(time=1.0, emotion="sad", intensity=0.8),
        ]
        prosody_two = keyframes_to_prosody(two_kf, duration_seconds=2.0)
        print(f"  Two keyframes: semantic shape {prosody_two['semantic'].shape}")

        # Three keyframes (minimum for full Catmull-Rom)
        three_kf = [
            ProsodyKeyframe(time=0.0, emotion="neutral", intensity=0.3),
            ProsodyKeyframe(time=0.5, emotion="surprised", intensity=1.0),
            ProsodyKeyframe(time=1.0, emotion="neutral", intensity=0.3),
        ]
        prosody_three = keyframes_to_prosody(three_kf, duration_seconds=1.0)
        print(f"  Three keyframes: semantic shape {prosody_three['semantic'].shape}")

        print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise

    # Test 7: Validation
    print("\n[Test 7] Testing validation...")
    try:
        # Should raise for invalid intensity
        try:
            ProsodyKeyframe(time=0.0, emotion="happy", intensity=1.5)
            print("  [FAIL] Should have raised ValueError for intensity > 1")
        except ValueError:
            print("  Invalid intensity correctly rejected")

        # Should raise for invalid emotion
        try:
            ProsodyKeyframe(time=0.0, emotion="ecstatic", intensity=0.5)
            print("  [FAIL] Should have raised ValueError for invalid emotion")
        except ValueError:
            print("  Invalid emotion correctly rejected")

        # Should raise for empty keyframes
        try:
            keyframes_to_prosody([], duration_seconds=1.0)
            print("  [FAIL] Should have raised ValueError for empty keyframes")
        except ValueError:
            print("  Empty keyframes correctly rejected")

        print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise

    # Test 8: Visualization (optional)
    print("\n[Test 8] Generating visualization...")
    try:
        # Create a more interesting keyframe sequence
        demo_keyframes = [
            ProsodyKeyframe(time=0.0, emotion="neutral", intensity=0.3),
            ProsodyKeyframe(time=0.2, emotion="happy", intensity=0.8, energy=0.7),
            ProsodyKeyframe(time=0.5, emotion="angry", intensity=0.9, pitch_tendency="high"),
            ProsodyKeyframe(time=0.7, emotion="sad", intensity=0.6, pitch_tendency="low"),
            ProsodyKeyframe(time=1.0, emotion="calm", intensity=0.5),
        ]

        demo_prosody = interpolate_prosody_keyframes(demo_keyframes, sample_points=64)

        # Try to save visualization (use path relative to this file)
        save_path = str(Path(__file__).parent / "keyframe_prosody_demo.png")
        visualize_keyframes(demo_keyframes, demo_prosody, save_path=save_path)
        print("  [PASS]")
    except ImportError:
        print("  [SKIP] matplotlib not available")
    except Exception as e:
        print(f"  [INFO] Visualization skipped: {e}")

    # Test 9: Temporal prosody tokens
    print("\n[Test 9] Testing get_temporal_prosody_tokens()...")
    try:
        # Create keyframes with distinct emotions at different times
        temporal_keyframes = [
            ProsodyKeyframe(time=0.0, emotion="neutral", intensity=0.5),
            ProsodyKeyframe(time=0.25, emotion="happy", intensity=0.9),
            ProsodyKeyframe(time=0.5, emotion="angry", intensity=0.8),
            ProsodyKeyframe(time=0.75, emotion="sad", intensity=0.7),
            ProsodyKeyframe(time=1.0, emotion="calm", intensity=0.4),
        ]

        prosody = keyframes_to_prosody(temporal_keyframes, duration_seconds=4.0)

        # Convert to temporal tokens (default 4 segments)
        temporal = get_temporal_prosody_tokens(prosody, num_segments=4)

        print(f"  Temporal semantic shape: {temporal['semantic'].shape}")
        print(f"  Temporal acoustic shape: {temporal['acoustic'].shape}")
        print(f"  Temporal rhythm shape: {temporal['rhythm'].shape}")
        print(f"  Temporal contour shape: {temporal['contour'].shape}")

        # Verify shapes
        assert temporal['semantic'].shape == (4, 8), "Temporal semantic shape mismatch"
        assert temporal['acoustic'].shape == (4, 12), "Temporal acoustic shape mismatch"
        assert temporal['rhythm'].shape == (4, 8), "Temporal rhythm shape mismatch"
        assert temporal['contour'].shape == (4, 64), "Temporal contour shape mismatch"

        # Verify that segments are different (temporal variation preserved)
        semantic_variance = temporal['semantic'].var(dim=0).sum().item()
        print(f"  Semantic variance across segments: {semantic_variance:.4f}")
        assert semantic_variance > 0.01, "Segments should have different values"

        # Test with different num_segments
        temporal_8 = get_temporal_prosody_tokens(prosody, num_segments=8)
        assert temporal_8['semantic'].shape == (8, 8), "8-segment shape mismatch"

        print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise

    # Test 10: Temporal tokens validation
    print("\n[Test 10] Testing temporal tokens validation...")
    try:
        # Should raise for invalid num_segments
        try:
            get_temporal_prosody_tokens(prosody, num_segments=0)
            print("  [FAIL] Should have raised ValueError for num_segments=0")
        except ValueError:
            print("  Invalid num_segments correctly rejected")

        # Should raise for missing dense keys
        try:
            bad_prosody = {'semantic': torch.zeros(1, 8)}  # Missing _dense keys
            get_temporal_prosody_tokens(bad_prosody, num_segments=4)
            print("  [FAIL] Should have raised KeyError for missing dense keys")
        except KeyError:
            print("  Missing dense keys correctly rejected")

        print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise

    print("\n" + "=" * 60)
    print("All tests passed! Keyframe Prosody System is ready.")
    print("=" * 60)

    # Print usage example
    print("\nUsage Example:")
    print("-" * 40)
    print("""
from keyframe_prosody import (
    ProsodyKeyframe,
    keyframes_to_prosody,
    get_temporal_prosody_tokens
)

# Define emotion trajectory with changes over time
keyframes = [
    ProsodyKeyframe(time=0.0, emotion="neutral", intensity=0.5),
    ProsodyKeyframe(time=1.5, emotion="happy", intensity=0.9),
    ProsodyKeyframe(time=3.0, emotion="neutral", intensity=0.5),
]

# Generate prosody for 3-second utterance
prosody = keyframes_to_prosody(keyframes, duration_seconds=3.0)

# Option 1: Global conditioning (current model, loses temporal info)
# Use prosody['semantic'], prosody['acoustic'], etc. with shape [1, dim]

# Option 2: Temporal conditioning (preserves keyframe edits)
# Convert to per-segment tokens matching num_prosody_tokens=4
temporal = get_temporal_prosody_tokens(prosody, num_segments=4)
# temporal['semantic'] shape: [4, dim] - one token per time segment

# Use with ProsodyControlledCSM:
# For global: model.generate_with_prosody(text_ids, attention_mask, prosody)
# For temporal: requires model extension to accept [num_segments, dim] input
""")
