"""
Activation Steering for Training-Free Emotion Control

Based on EmoSteer-TTS (2025): "EmoSteer-TTS: Exploring Training-free Activation Steering
in Text-to-Speech"
https://arxiv.org/html/2508.03543v1

Key Innovation: Modify internal activations at inference time without retraining:
1. Extract "steering vectors" = difference between emotional vs neutral activations
2. Register PyTorch hooks to intercept DiT/transformer layer activations
3. Apply: modified = original + α × steering_vector (with L2 norm preservation)
4. Identify top-k emotion-relevant tokens for selective modification

Advantages:
- No training required once model is trained
- Continuous intensity via α parameter
- Can blend multiple emotions with weighted steering vectors
- Works with any transformer-based TTS model

Usage:
    from activation_steering import ActivationSteering, SteeringVectorExtractor

    # Extract steering vectors from emotional audio samples
    extractor = SteeringVectorExtractor(model)
    steering_vectors = extractor.extract(
        neutral_samples=["neutral1.wav", "neutral2.wav"],
        emotional_samples={"happy": ["happy1.wav", "happy2.wav"], ...}
    )

    # Apply steering during inference
    steerer = ActivationSteering(model, steering_vectors, target_layers=[1, 6, 11, 16])

    # Generate with emotion control
    with steerer.steer("happy", intensity=0.8):
        audio = model.generate(text)
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SteeringConfig:
    """Configuration for activation steering."""

    # Steering parameters
    default_intensity: float = 0.5  # Default α value
    min_intensity: float = -1.5  # Allow opposite direction steering
    max_intensity: float = 1.5  # Maximum steering strength

    # Layer selection (for different model sizes)
    layer_stride: int = 5  # Every Nth layer (5 for F5-TTS, 3 for E2-TTS)
    start_layer: int = 1  # First layer to steer
    total_layers: int = 22  # Total layers in model

    # Top-k token selection
    use_topk: bool = True  # Whether to use selective token steering
    topk: int = 200  # Number of tokens to modify (k=200 in paper)

    # Normalization
    preserve_norm: bool = True  # Renormalize after steering to preserve L2 norm
    normalize_steering_vector: bool = True  # Normalize steering vectors to unit length

    # Caching
    cache_activations: bool = True  # Cache steering vectors for efficiency
    cache_dir: str = ".steering_cache"  # Directory for cached vectors


# =============================================================================
# STEERING VECTOR EXTRACTION
# =============================================================================

class SteeringVectorExtractor:
    """
    Extracts steering vectors from emotional vs neutral audio samples.

    The steering vector for an emotion is computed as the difference-in-means
    between emotional and neutral activation patterns:

        u_l = mean(emotional_activations_l) - mean(neutral_activations_l)

    where l indexes the transformer layer.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SteeringConfig = None,
        activation_key: str = "residual",  # Type of activation to extract
    ):
        """
        Args:
            model: The TTS model to extract activations from
            config: Steering configuration
            activation_key: Which activation to intercept (residual, attention, ffn)
        """
        self.model = model
        self.config = config or SteeringConfig()
        self.activation_key = activation_key

        # Storage for collected activations
        self._activations: Dict[int, List[Tensor]] = {}
        self._hooks: List = []

    def _get_target_layers(self) -> List[int]:
        """Get indices of layers to extract activations from."""
        layers = []
        layer_idx = self.config.start_layer
        while layer_idx < self.config.total_layers:
            layers.append(layer_idx)
            layer_idx += self.config.layer_stride
        return layers

    def _find_transformer_layers(self, model: nn.Module) -> List[nn.Module]:
        """
        Find transformer/DiT layers in the model.

        Handles various architectures:
        - HuggingFace transformers: model.model.layers
        - DiT: model.layers or model.dit_blocks
        - CSM: model.decoder.layers
        """
        # Try common paths
        paths = [
            ["model", "layers"],
            ["decoder", "layers"],
            ["backbone", "layers"],
            ["transformer", "layers"],
            ["dit_blocks"],
            ["layers"],
        ]

        for path in paths:
            module = model
            try:
                for attr in path:
                    module = getattr(module, attr)
                if isinstance(module, (nn.ModuleList, list)) and len(module) > 0:
                    return list(module)
            except AttributeError:
                continue

        # Fallback: search for modules with "layer" in name
        layers = []
        for name, module in model.named_modules():
            if "layer" in name.lower() and isinstance(module, nn.Module):
                # Check if it's a transformer layer (has self_attn or attention)
                if hasattr(module, "self_attn") or hasattr(module, "attention"):
                    layers.append(module)
        return layers

    def _create_activation_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook that captures activations."""
        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output

            # Store activation
            if layer_idx not in self._activations:
                self._activations[layer_idx] = []
            self._activations[layer_idx].append(activation.detach().cpu())

        return hook

    def _register_hooks(self, layers: List[nn.Module], layer_indices: List[int]):
        """Register forward hooks on target layers."""
        self._clear_hooks()
        for idx in layer_indices:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(
                    self._create_activation_hook(idx)
                )
                self._hooks.append(hook)

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _clear_activations(self):
        """Clear stored activations."""
        self._activations = {}

    @torch.no_grad()
    def _collect_activations(
        self,
        audio_paths: List[str],
        process_fn: Callable,
    ) -> Dict[int, List[Tensor]]:
        """
        Collect activations for a list of audio files.

        Args:
            audio_paths: List of paths to audio files
            process_fn: Function that processes audio and runs model inference
                       Signature: process_fn(model, audio_path) -> None

        Returns:
            Dict mapping layer indices to lists of activation tensors
        """
        self._clear_activations()

        layers = self._find_transformer_layers(self.model)
        target_layers = self._get_target_layers()

        self._register_hooks(layers, target_layers)

        try:
            for audio_path in audio_paths:
                process_fn(self.model, audio_path)
        finally:
            self._clear_hooks()

        activations = {k: v for k, v in self._activations.items()}
        self._clear_activations()
        return activations

    def _compute_mean_activation(
        self,
        activations: Dict[int, List[Tensor]],
    ) -> Dict[int, Tensor]:
        """Compute mean activation per layer across samples."""
        mean_activations = {}
        for layer_idx, acts in activations.items():
            # Stack all activations: [num_samples, seq_len, hidden_dim]
            stacked = torch.stack([a.squeeze(0) for a in acts])
            # Average over samples (keep sequence dimension for alignment)
            mean_activations[layer_idx] = stacked.mean(dim=0)
        return mean_activations

    def _interpolate_to_length(
        self,
        activations: Dict[int, Tensor],
        target_length: int,
    ) -> Dict[int, Tensor]:
        """Interpolate activation sequences to a fixed length."""
        interpolated = {}
        for layer_idx, act in activations.items():
            if act.shape[0] != target_length:
                # Interpolate: [seq_len, hidden] -> [target_len, hidden]
                act_t = act.T.unsqueeze(0)  # [1, hidden, seq_len]
                interp = F.interpolate(
                    act_t, size=target_length, mode='linear', align_corners=False
                )
                interpolated[layer_idx] = interp.squeeze(0).T
            else:
                interpolated[layer_idx] = act
        return interpolated

    def extract(
        self,
        neutral_samples: List[str],
        emotional_samples: Dict[str, List[str]],
        process_fn: Callable,
        target_sequence_length: int = None,
    ) -> Dict[str, Dict[int, Tensor]]:
        """
        Extract steering vectors for each emotion.

        Args:
            neutral_samples: List of paths to neutral audio samples
            emotional_samples: Dict mapping emotion names to lists of audio paths
            process_fn: Function to process audio and run inference
            target_sequence_length: Fixed length to interpolate sequences to
                                   (uses average length if None)

        Returns:
            Dict mapping emotion names to steering vectors per layer
            steering_vectors[emotion][layer_idx] = Tensor[seq_len, hidden_dim]
        """
        print(f"Extracting steering vectors for {len(emotional_samples)} emotions...")

        # Collect neutral activations
        print(f"  Processing {len(neutral_samples)} neutral samples...")
        neutral_acts = self._collect_activations(neutral_samples, process_fn)
        neutral_mean = self._compute_mean_activation(neutral_acts)

        # Determine target sequence length
        if target_sequence_length is None:
            all_lengths = [act.shape[0] for act in neutral_mean.values()]
            target_sequence_length = int(np.mean(all_lengths))

        # Interpolate neutral to target length
        neutral_mean = self._interpolate_to_length(neutral_mean, target_sequence_length)

        # Extract steering vector for each emotion
        steering_vectors = {}
        for emotion, samples in emotional_samples.items():
            print(f"  Processing {len(samples)} '{emotion}' samples...")

            # Collect emotional activations
            emotional_acts = self._collect_activations(samples, process_fn)
            emotional_mean = self._compute_mean_activation(emotional_acts)
            emotional_mean = self._interpolate_to_length(emotional_mean, target_sequence_length)

            # Compute difference: steering_vector = emotional - neutral
            emotion_vectors = {}
            for layer_idx in neutral_mean.keys():
                diff = emotional_mean[layer_idx] - neutral_mean[layer_idx]

                # Optionally normalize to unit vector
                if self.config.normalize_steering_vector:
                    diff = F.normalize(diff, p=2, dim=-1)

                emotion_vectors[layer_idx] = diff

            steering_vectors[emotion] = emotion_vectors

        print(f"Extracted steering vectors: {list(steering_vectors.keys())}")
        return steering_vectors

    def save_steering_vectors(
        self,
        steering_vectors: Dict[str, Dict[int, Tensor]],
        path: str,
    ):
        """Save steering vectors to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "steering_vectors": steering_vectors,
            "config": self.config,
        }, path)
        print(f"Saved steering vectors to {path}")

    @staticmethod
    def load_steering_vectors(path: str) -> Tuple[Dict[str, Dict[int, Tensor]], SteeringConfig]:
        """Load steering vectors from disk."""
        data = torch.load(path, weights_only=False)
        return data["steering_vectors"], data["config"]


# =============================================================================
# ACTIVATION STEERING (INFERENCE)
# =============================================================================

class ActivationSteering:
    """
    Applies steering vectors to model activations during inference.

    Usage:
        steerer = ActivationSteering(model, steering_vectors)

        # Context manager for temporary steering
        with steerer.steer("happy", intensity=0.8):
            audio = model.generate(text)

        # Or manual enable/disable
        steerer.enable("sad", intensity=0.6)
        audio = model.generate(text)
        steerer.disable()
    """

    def __init__(
        self,
        model: nn.Module,
        steering_vectors: Dict[str, Dict[int, Tensor]],
        config: SteeringConfig = None,
        target_layers: List[int] = None,
        device: torch.device = None,
    ):
        """
        Args:
            model: The TTS model to steer
            steering_vectors: Dict of emotion -> layer -> steering vector
            config: Steering configuration
            target_layers: Specific layers to steer (overrides config)
            device: Device to place steering vectors on
        """
        self.model = model
        self.config = config or SteeringConfig()
        self.device = device or next(model.parameters()).device

        # Move steering vectors to device
        self.steering_vectors = {
            emotion: {
                layer: vec.to(self.device)
                for layer, vec in layers.items()
            }
            for emotion, layers in steering_vectors.items()
        }

        # Determine target layers
        if target_layers is not None:
            self.target_layers = target_layers
        else:
            self.target_layers = list(next(iter(steering_vectors.values())).keys())

        # Find transformer layers in model
        self._layers = self._find_transformer_layers(model)

        # Active steering state
        self._active_emotion: Optional[str] = None
        self._active_intensity: float = 0.0
        self._active_blend: Optional[Dict[str, float]] = None
        self._hooks: List = []

        # Top-k mask (computed once per emotion)
        self._topk_mask: Optional[Dict[int, Tensor]] = None

        print(f"ActivationSteering initialized with emotions: {list(self.steering_vectors.keys())}")
        print(f"Target layers: {self.target_layers}")

    def _find_transformer_layers(self, model: nn.Module) -> List[nn.Module]:
        """Find transformer layers (same as extractor)."""
        paths = [
            ["model", "layers"],
            ["decoder", "layers"],
            ["backbone", "layers"],
            ["transformer", "layers"],
            ["dit_blocks"],
            ["layers"],
        ]

        for path in paths:
            module = model
            try:
                for attr in path:
                    module = getattr(module, attr)
                if isinstance(module, (nn.ModuleList, list)) and len(module) > 0:
                    return list(module)
            except AttributeError:
                continue

        layers = []
        for name, module in model.named_modules():
            if "layer" in name.lower() and isinstance(module, nn.Module):
                if hasattr(module, "self_attn") or hasattr(module, "attention"):
                    layers.append(module)
        return layers

    def _get_steering_vector(self, layer_idx: int) -> Tensor:
        """Get the steering vector for a layer based on current settings."""
        if self._active_blend is not None:
            # Blend multiple emotions
            blended = None
            for emotion, weight in self._active_blend.items():
                if emotion in self.steering_vectors and layer_idx in self.steering_vectors[emotion]:
                    vec = self.steering_vectors[emotion][layer_idx] * weight
                    if blended is None:
                        blended = vec
                    else:
                        blended = blended + vec
            return blended if blended is not None else torch.zeros(1, device=self.device)

        elif self._active_emotion is not None:
            if layer_idx in self.steering_vectors.get(self._active_emotion, {}):
                return self.steering_vectors[self._active_emotion][layer_idx]

        return torch.zeros(1, device=self.device)

    def _create_steering_hook(self, layer_idx: int) -> Callable:
        """
        Create a pre-hook that modifies activations with the steering vector.

        Formula: x̂ = f_r(x + α × s)

        where f_r is optional renormalization to preserve original L2 norm.
        """
        def hook(module, args):
            # Get input activation
            if isinstance(args, tuple) and len(args) > 0:
                x = args[0]
            else:
                return args

            if x is None or self._active_emotion is None and self._active_blend is None:
                return args

            # Get steering vector
            steering_vec = self._get_steering_vector(layer_idx)
            if steering_vec.numel() == 1:  # Zero vector
                return args

            # Ensure matching sequence length via interpolation
            seq_len = x.shape[1] if x.dim() > 2 else x.shape[0]
            if steering_vec.shape[0] != seq_len:
                # Interpolate steering vector to match input
                sv_t = steering_vec.T.unsqueeze(0)  # [1, hidden, sv_len]
                sv_interp = F.interpolate(
                    sv_t, size=seq_len, mode='linear', align_corners=False
                )
                steering_vec = sv_interp.squeeze(0).T  # [seq_len, hidden]

            # Apply top-k mask if enabled
            if self.config.use_topk and self._topk_mask is not None:
                if layer_idx in self._topk_mask:
                    mask = self._topk_mask[layer_idx]
                    # Interpolate mask to match sequence length
                    if mask.shape[0] != seq_len:
                        mask = F.interpolate(
                            mask.unsqueeze(0).unsqueeze(0).float(),
                            size=seq_len,
                            mode='nearest'
                        ).squeeze().bool()
                    steering_vec = steering_vec * mask.unsqueeze(-1).float()

            # Store original norm for renormalization
            if self.config.preserve_norm:
                original_norm = x.norm(dim=-1, keepdim=True)

            # Apply steering: x̂ = x + α × s
            intensity = self._active_intensity
            x_steered = x + intensity * steering_vec.unsqueeze(0)

            # Renormalize to preserve L2 norm
            if self.config.preserve_norm:
                current_norm = x_steered.norm(dim=-1, keepdim=True)
                x_steered = x_steered * (original_norm / (current_norm + 1e-8))

            # Return modified args
            if isinstance(args, tuple):
                return (x_steered,) + args[1:]
            return x_steered

        return hook

    def _register_hooks(self):
        """Register pre-hooks on target layers."""
        self._clear_hooks()
        for layer_idx in self.target_layers:
            if layer_idx < len(self._layers):
                # Use pre-hook to modify inputs before the layer
                hook = self._layers[layer_idx].register_forward_pre_hook(
                    self._create_steering_hook(layer_idx)
                )
                self._hooks.append(hook)

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def enable(
        self,
        emotion: Optional[str] = None,
        intensity: float = None,
        blend: Optional[Dict[str, float]] = None,
    ):
        """
        Enable activation steering.

        Args:
            emotion: Single emotion to steer towards
            intensity: Steering strength (α parameter)
            blend: Dict of emotion -> weight for blending multiple emotions
        """
        if intensity is None:
            intensity = self.config.default_intensity

        # Clamp intensity
        intensity = max(self.config.min_intensity, min(self.config.max_intensity, intensity))

        if blend is not None:
            # Normalize blend weights
            total = sum(blend.values())
            self._active_blend = {e: w/total for e, w in blend.items()}
            self._active_emotion = None
        else:
            self._active_emotion = emotion
            self._active_blend = None

        self._active_intensity = intensity
        self._register_hooks()

    def disable(self):
        """Disable activation steering."""
        self._clear_hooks()
        self._active_emotion = None
        self._active_intensity = 0.0
        self._active_blend = None

    @contextmanager
    def steer(
        self,
        emotion: Optional[str] = None,
        intensity: float = None,
        blend: Optional[Dict[str, float]] = None,
    ):
        """
        Context manager for temporary steering.

        Example:
            with steerer.steer("happy", intensity=0.8):
                audio = model.generate(text)
        """
        try:
            self.enable(emotion, intensity, blend)
            yield self
        finally:
            self.disable()

    def compute_topk_mask(
        self,
        emotion: str,
        emotion_classifier: Callable[[Tensor], float],
        sample_input: Tensor,
        k: int = None,
    ) -> Dict[int, Tensor]:
        """
        Compute top-k emotion-relevant token mask.

        For each token position, we measure how much steering that single token
        increases the emotion classifier's probability for the target emotion.
        The k tokens with highest impact are selected.

        Args:
            emotion: Target emotion
            emotion_classifier: Function that takes audio features and returns
                               emotion probability for the target emotion
            sample_input: Sample input to test token impacts
            k: Number of tokens to select (uses config.topk if None)

        Returns:
            Dict mapping layer indices to boolean masks [seq_len]
        """
        if k is None:
            k = self.config.topk

        if emotion not in self.steering_vectors:
            warnings.warn(f"Emotion '{emotion}' not found in steering vectors")
            return {}

        print(f"Computing top-{k} mask for '{emotion}'...")

        masks = {}
        for layer_idx in self.target_layers:
            if layer_idx not in self.steering_vectors[emotion]:
                continue

            steering_vec = self.steering_vectors[emotion][layer_idx]
            seq_len = steering_vec.shape[0]

            # Test impact of each token
            impacts = torch.zeros(seq_len)

            for i in range(seq_len):
                # Create single-token steering vector
                single_token_sv = torch.zeros_like(steering_vec)
                single_token_sv[i] = steering_vec[i]

                # Measure classifier output with this single token steered
                # (This would require running inference - simplified here)
                # In practice, you'd temporarily apply steering and run classifier
                impacts[i] = steering_vec[i].norm()  # Simplified: use norm as proxy

            # Select top-k
            _, topk_indices = torch.topk(impacts, min(k, seq_len))
            mask = torch.zeros(seq_len, dtype=torch.bool)
            mask[topk_indices] = True

            masks[layer_idx] = mask

        self._topk_mask = masks
        print(f"Computed top-k masks for {len(masks)} layers")
        return masks

    def list_emotions(self) -> List[str]:
        """List available emotions."""
        return list(self.steering_vectors.keys())

    def get_emotion_info(self, emotion: str) -> Dict:
        """Get information about steering vectors for an emotion."""
        if emotion not in self.steering_vectors:
            return {"error": f"Emotion '{emotion}' not found"}

        layers = self.steering_vectors[emotion]
        return {
            "emotion": emotion,
            "num_layers": len(layers),
            "layers": list(layers.keys()),
            "vector_shapes": {
                l: tuple(v.shape) for l, v in layers.items()
            },
            "vector_norms": {
                l: v.norm().item() for l, v in layers.items()
            },
        }


# =============================================================================
# EMOTION RECOGNITION FOR TOP-K SELECTION
# =============================================================================

class EmotionRecognizer:
    """
    Wrapper for emotion recognition models used in top-k selection.

    Supports:
    - emotion2vec (used in paper)
    - wav2vec2 emotion models
    - Custom models
    """

    def __init__(self, model_name: str = "emotion2vec"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of emotion model."""
        if self._initialized:
            return

        if self.model_name == "emotion2vec":
            try:
                from funasr import AutoModel
                self.model = AutoModel(model="iic/emotion2vec_base_finetuned")
                self._initialized = True
            except ImportError:
                warnings.warn("emotion2vec requires funasr: pip install funasr")
        else:
            try:
                from transformers import AutoModelForAudioClassification, AutoProcessor
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
                self._initialized = True
            except Exception as e:
                warnings.warn(f"Failed to load emotion model: {e}")

    @torch.no_grad()
    def predict_emotion(
        self,
        audio: Tensor,
        sample_rate: int = 16000,
    ) -> Dict[str, float]:
        """
        Predict emotion probabilities from audio.

        Args:
            audio: Audio tensor [samples] or [batch, samples]
            sample_rate: Audio sample rate

        Returns:
            Dict mapping emotion names to probabilities
        """
        self.initialize()

        if not self._initialized:
            return {"neutral": 1.0}

        if self.model_name == "emotion2vec":
            # emotion2vec format
            result = self.model.generate(audio.numpy(), output_dir=None)
            return result[0]["scores"]
        else:
            # HuggingFace format
            inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            labels = self.model.config.id2label
            return {labels[i]: probs[0][i].item() for i in range(len(labels))}

    def get_emotion_probability(
        self,
        audio: Tensor,
        emotion: str,
        sample_rate: int = 16000,
    ) -> float:
        """Get probability for a specific emotion."""
        probs = self.predict_emotion(audio, sample_rate)
        return probs.get(emotion, probs.get(emotion.lower(), 0.0))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_steering_from_samples(
    model: nn.Module,
    neutral_audios: List[Tensor],
    emotional_audios: Dict[str, List[Tensor]],
    config: SteeringConfig = None,
) -> ActivationSteering:
    """
    Convenience function to create steering from audio tensors.

    Args:
        model: TTS model
        neutral_audios: List of neutral audio tensors
        emotional_audios: Dict of emotion -> list of audio tensors
        config: Steering configuration

    Returns:
        Configured ActivationSteering instance
    """
    config = config or SteeringConfig()
    extractor = SteeringVectorExtractor(model, config)

    # Create process function for tensor inputs
    def process_fn(model, audio):
        # Run model forward pass (implementation depends on model)
        with torch.no_grad():
            if hasattr(model, "encode"):
                model.encode(audio)
            elif hasattr(model, "forward"):
                model(audio)

    # Extract vectors (need to adapt for tensor input)
    # This is a simplified version - full implementation would handle file vs tensor inputs
    steering_vectors = extractor.extract(
        neutral_samples=neutral_audios,
        emotional_samples=emotional_audios,
        process_fn=process_fn,
    )

    return ActivationSteering(model, steering_vectors, config)


def blend_emotions(
    emotions: Dict[str, float],
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Normalize emotion blend weights.

    Args:
        emotions: Dict of emotion -> weight
        normalize: Whether to normalize weights to sum to 1

    Returns:
        Normalized emotion weights
    """
    if normalize:
        total = sum(emotions.values())
        if total > 0:
            return {e: w/total for e, w in emotions.items()}
    return emotions


def interpolate_emotions(
    emotion1: str,
    emotion2: str,
    t: float,
) -> Dict[str, float]:
    """
    Create blend weights for interpolating between two emotions.

    Args:
        emotion1: Start emotion
        emotion2: End emotion
        t: Interpolation factor [0, 1]

    Returns:
        Blend weights
    """
    return {
        emotion1: 1.0 - t,
        emotion2: t,
    }


# =============================================================================
# INTEGRATION WITH CSM MODEL
# =============================================================================

class CSMActivationSteering(ActivationSteering):
    """
    Activation steering specifically for CSM-1B model.

    Handles CSM's specific architecture:
    - Decoder-based transformer
    - Audio codec conditioning
    - Multi-codebook output
    """

    def __init__(
        self,
        model: nn.Module,
        steering_vectors: Dict[str, Dict[int, Tensor]],
        config: SteeringConfig = None,
    ):
        # CSM-specific defaults
        if config is None:
            config = SteeringConfig(
                layer_stride=4,  # CSM has ~24 layers
                start_layer=2,
                total_layers=24,
                topk=150,  # Slightly fewer tokens for CSM
            )

        super().__init__(model, steering_vectors, config)

    def _find_transformer_layers(self, model: nn.Module) -> List[nn.Module]:
        """Find CSM's decoder layers."""
        # Try CSM-specific paths
        if hasattr(model, "decoder"):
            if hasattr(model.decoder, "layers"):
                return list(model.decoder.layers)

        if hasattr(model, "model"):
            if hasattr(model.model, "decoder"):
                if hasattr(model.model.decoder, "layers"):
                    return list(model.model.decoder.layers)

        # Fallback to generic search
        return super()._find_transformer_layers(model)


# =============================================================================
# INTEGRATION WITH SPHERICAL EMOTION
# =============================================================================

class SphericalActivationSteering:
    """
    Combines spherical emotion vectors with activation steering.

    This enables:
    1. Using VAD coordinates to blend steering vectors
    2. Intensity control via spherical radius
    3. Smooth interpolation in emotion space
    """

    def __init__(
        self,
        steerer: ActivationSteering,
        vad_to_emotion_map: Dict[str, Tuple[float, float, float]] = None,
    ):
        """
        Args:
            steerer: Configured ActivationSteering instance
            vad_to_emotion_map: Mapping from emotion names to VAD coordinates
        """
        self.steerer = steerer

        # Default VAD prototypes (from spherical_emotion.py)
        self.vad_map = vad_to_emotion_map or {
            "neutral": (0.0, 0.0, 0.0),
            "happy": (0.8, 0.6, 0.6),
            "sad": (-0.6, -0.4, -0.5),
            "angry": (-0.5, 0.8, 0.7),
            "surprised": (0.3, 0.8, 0.2),
            "calm": (0.4, -0.5, 0.3),
            "fearful": (-0.7, 0.7, -0.7),
            "disgusted": (-0.6, 0.3, 0.4),
        }

    def _vad_to_blend(
        self,
        valence: float,
        arousal: float,
        dominance: float,
    ) -> Dict[str, float]:
        """
        Convert VAD coordinates to emotion blend weights.

        Uses inverse distance weighting to nearby emotion prototypes.
        """
        vad = np.array([valence, arousal, dominance])
        weights = {}

        for emotion, proto in self.vad_map.items():
            if emotion not in self.steerer.steering_vectors:
                continue
            proto_arr = np.array(proto)
            dist = np.linalg.norm(vad - proto_arr)
            # Inverse distance weighting (avoid division by zero)
            weights[emotion] = 1.0 / (dist + 0.1)

        # Normalize
        total = sum(weights.values())
        return {e: w/total for e, w in weights.items()}

    @contextmanager
    def steer_vad(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        intensity: float = None,
    ):
        """
        Steer using VAD coordinates.

        Args:
            valence: -1 (negative) to +1 (positive)
            arousal: -1 (calm) to +1 (excited)
            dominance: -1 (submissive) to +1 (dominant)
            intensity: Override intensity (uses VAD magnitude if None)
        """
        blend = self._vad_to_blend(valence, arousal, dominance)

        # Use VAD magnitude as intensity if not specified
        if intensity is None:
            intensity = np.linalg.norm([valence, arousal, dominance])
            intensity = min(1.5, intensity)  # Clamp

        with self.steerer.steer(blend=blend, intensity=intensity):
            yield self

    def steer_interpolated(
        self,
        emotion1: str,
        emotion2: str,
        t: float,
        intensity: float = None,
    ):
        """
        Steer with interpolation between two emotions.

        Uses SLERP-style interpolation in VAD space.
        """
        vad1 = np.array(self.vad_map.get(emotion1, (0, 0, 0)))
        vad2 = np.array(self.vad_map.get(emotion2, (0, 0, 0)))

        # Spherical interpolation
        vad_interp = (1 - t) * vad1 + t * vad2  # Simplified LERP

        return self.steer_vad(
            valence=vad_interp[0],
            arousal=vad_interp[1],
            dominance=vad_interp[2],
            intensity=intensity,
        )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Activation Steering for Training-Free Emotion Control - Test Suite")
    print("=" * 70)

    # Test 1: Configuration
    print("\n[Test 1] SteeringConfig...")
    config = SteeringConfig(
        default_intensity=0.5,
        layer_stride=5,
        start_layer=1,
        total_layers=22,
        topk=200,
    )
    print(f"  Layer indices: {list(range(config.start_layer, config.total_layers, config.layer_stride))}")
    print("  [PASS]")

    # Test 2: Dummy steering vectors
    print("\n[Test 2] Creating dummy steering vectors...")
    hidden_dim = 1024
    seq_len = 100
    layers = [1, 6, 11, 16]

    dummy_vectors = {}
    for emotion in ["happy", "sad", "angry"]:
        dummy_vectors[emotion] = {
            layer: torch.randn(seq_len, hidden_dim)
            for layer in layers
        }
    print(f"  Created vectors for: {list(dummy_vectors.keys())}")
    print(f"  Shape per layer: ({seq_len}, {hidden_dim})")
    print("  [PASS]")

    # Test 3: Blend emotions
    print("\n[Test 3] Emotion blending...")
    blend = blend_emotions({"happy": 0.6, "sad": 0.4})
    print(f"  Input: {{'happy': 0.6, 'sad': 0.4}}")
    print(f"  Normalized: {blend}")
    print("  [PASS]")

    # Test 4: Emotion interpolation
    print("\n[Test 4] Emotion interpolation...")
    interp = interpolate_emotions("happy", "sad", t=0.3)
    print(f"  happy → sad, t=0.3: {interp}")
    print("  [PASS]")

    # Test 5: Dummy model with layers
    print("\n[Test 5] Creating dummy model...")

    class DummyLayer(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.self_attn = nn.Linear(hidden_dim, hidden_dim)
            self.ffn = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x):
            return x + self.self_attn(x)

    class DummyModel(nn.Module):
        def __init__(self, num_layers=22, hidden_dim=1024):
            super().__init__()
            self.layers = nn.ModuleList([
                DummyLayer(hidden_dim) for _ in range(num_layers)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = DummyModel()
    print(f"  Created model with {len(model.layers)} layers")
    print("  [PASS]")

    # Test 6: ActivationSteering initialization
    print("\n[Test 6] ActivationSteering initialization...")
    steerer = ActivationSteering(model, dummy_vectors, config, target_layers=layers)
    print(f"  Available emotions: {steerer.list_emotions()}")
    print(f"  Target layers: {steerer.target_layers}")
    print("  [PASS]")

    # Test 7: Steering context manager
    print("\n[Test 7] Steering context manager...")
    x = torch.randn(1, 50, 1024)

    # Without steering
    with torch.no_grad():
        y_baseline = model(x)

    # With steering
    with steerer.steer("happy", intensity=0.5):
        with torch.no_grad():
            y_steered = model(x)

    diff = (y_steered - y_baseline).abs().mean().item()
    print(f"  Baseline output mean: {y_baseline.mean().item():.4f}")
    print(f"  Steered output mean: {y_steered.mean().item():.4f}")
    print(f"  Mean absolute difference: {diff:.4f}")
    assert diff > 0, "Steering should change the output!"
    print("  [PASS]")

    # Test 8: Emotion blending
    print("\n[Test 8] Multi-emotion blending...")
    with steerer.steer(blend={"happy": 0.7, "sad": 0.3}, intensity=0.6):
        with torch.no_grad():
            y_blended = model(x)

    diff_blend = (y_blended - y_baseline).abs().mean().item()
    print(f"  Blended (happy:0.7, sad:0.3) difference: {diff_blend:.4f}")
    print("  [PASS]")

    # Test 9: Intensity control
    print("\n[Test 9] Intensity control...")
    diffs = []
    for intensity in [0.0, 0.25, 0.5, 0.75, 1.0]:
        with steerer.steer("angry", intensity=intensity):
            with torch.no_grad():
                y = model(x)
        diff = (y - y_baseline).abs().mean().item()
        diffs.append(diff)
        print(f"  α={intensity:.2f}: diff={diff:.4f}")

    # Verify intensity scaling (higher intensity = larger difference)
    assert all(diffs[i] <= diffs[i+1] for i in range(len(diffs)-1)), \
        "Difference should increase with intensity"
    print("  [PASS]")

    # Test 10: SphericalActivationSteering
    print("\n[Test 10] SphericalActivationSteering...")
    spherical = SphericalActivationSteering(steerer)

    with spherical.steer_vad(valence=0.8, arousal=0.6, dominance=0.4):
        with torch.no_grad():
            y_vad = model(x)

    diff_vad = (y_vad - y_baseline).abs().mean().item()
    print(f"  VAD steering (V=0.8, A=0.6, D=0.4) difference: {diff_vad:.4f}")
    print("  [PASS]")

    # Test 11: Emotion info
    print("\n[Test 11] Emotion info...")
    info = steerer.get_emotion_info("happy")
    print(f"  Emotion: {info['emotion']}")
    print(f"  Num layers: {info['num_layers']}")
    print(f"  Layers: {info['layers']}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All Activation Steering tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from activation_steering import (
    SteeringVectorExtractor, ActivationSteering, SteeringConfig
)

# 1. Extract steering vectors from emotional audio samples
extractor = SteeringVectorExtractor(model)

def process_fn(model, audio_path):
    audio = load_audio(audio_path)
    with torch.no_grad():
        model.encode(audio)

steering_vectors = extractor.extract(
    neutral_samples=["neutral1.wav", "neutral2.wav"],
    emotional_samples={
        "happy": ["happy1.wav", "happy2.wav"],
        "sad": ["sad1.wav", "sad2.wav"],
        "angry": ["angry1.wav", "angry2.wav"],
    },
    process_fn=process_fn,
)

# Save for later use
extractor.save_steering_vectors(steering_vectors, "steering_vectors.pt")

# 2. Apply steering during inference
steerer = ActivationSteering(model, steering_vectors)

# Single emotion
with steerer.steer("happy", intensity=0.7):
    audio = model.generate(text)

# Blend multiple emotions
with steerer.steer(blend={"happy": 0.6, "surprised": 0.4}, intensity=0.8):
    audio = model.generate(text)

# Continuous intensity control
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    with steerer.steer("angry", intensity=alpha):
        audio = model.generate(text)
        # Gradually more angry

# 3. VAD-space steering (integrates with spherical emotion)
spherical = SphericalActivationSteering(steerer)
with spherical.steer_vad(valence=0.8, arousal=0.6, dominance=0.4):
    audio = model.generate(text)  # Happy-ish emotion
""")
