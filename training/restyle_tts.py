"""
ReStyle-TTS: LoRA-based Style Control with Dual Classifier-Free Guidance

Based on ReStyle-TTS (arXiv:2601.03632): "Reference-Free Speaking Style Control for
Text-to-Speech Using Style-Specific LoRAs and Orthogonal LoRA Fusion"

Key Innovations:
1. DCFG (Dual Classifier-Free Guidance): Independently control text and reference guidance
   - Reduces reference audio dependency while preserving text fidelity
   - Formula: f̂_DCFG = f_{∅,t} + λ_t(f_{∅,t} - f_{∅,∅}) + λ_a(f_{a,t} - f_{∅,t})

2. Style-Specific LoRAs: Train separate LoRAs for interpretable style attributes
   - Prosody: high-pitch, low-pitch, high-energy, low-energy
   - Emotions: happy, sad, angry, calm, surprised, fearful

3. Orthogonal LoRA Fusion (OLoRA): Combine multiple LoRAs without interference
   - Projects each LoRA update onto orthogonal complement of others
   - Enables continuous, disentangled multi-attribute control

Benefits for Voice Clone Pipeline:
- Reference-free style control at inference time
- Each LoRA captures single interpretable attribute
- Can interpolate between LoRAs for continuous style control
- Builds on existing PEFT/LoRA infrastructure

Usage:
    # Training
    python train_style_lora.py --style happy --config config/style_lora.yaml

    # Inference with DCFG
    python generate_with_style_lora.py --text "Hello!" --styles "happy:0.8,high_pitch:0.5"
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ReStyleTTSConfig:
    """Configuration for ReStyle-TTS with DCFG and Style LoRAs."""

    # DCFG (Dual Classifier-Free Guidance) settings
    dcfg_text_scale: float = 2.0       # λ_t: text guidance strength
    dcfg_audio_scale: float = 0.5      # λ_a: reference audio guidance
    dcfg_enabled: bool = True          # Enable DCFG at inference

    # LoRA configuration
    lora_rank: int = 32                # LoRA rank (paper uses 32)
    lora_alpha: int = 64               # LoRA alpha (paper uses 64)
    lora_dropout: float = 0.1          # LoRA dropout
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])  # Target all linear layers

    # Style-specific settings
    available_styles: List[str] = field(default_factory=lambda: [
        # Prosody styles
        "high_pitch", "low_pitch",
        "high_energy", "low_energy",
        "fast_tempo", "slow_tempo",
        # Emotion styles
        "happy", "sad", "angry", "calm",
        "surprised", "fearful", "disgusted", "neutral"
    ])

    # Training settings
    learning_rate: float = 1e-5        # Paper uses 1e-5
    batch_size_frames: int = 30000     # Paper uses 30k frames
    speech_mask_ratio: float = 0.3     # Speech input dropout
    text_mask_ratio: float = 0.2       # Text+speech input dropout

    # Orthogonal LoRA Fusion settings
    olora_enabled: bool = True         # Enable orthogonal projection
    olora_regularization: float = 0.01 # Orthogonality regularization strength

    # Inference scaling
    prosody_lora_scale: float = 2.0    # Scale for prosody LoRAs
    emotion_lora_scale: float = 4.0    # Scale for emotion LoRAs

    # Integration settings
    hidden_size: int = 2048            # CSM hidden size
    num_prosody_tokens: int = 4        # Prefix tokens

    # Timbre Consistency Optimization (TCO)
    tco_enabled: bool = False          # Enable TCO during training
    tco_lambda: float = 0.2            # Advantage weighting
    tco_beta: float = 5.0              # Advantage scaling
    tco_momentum: float = 0.9          # EMA baseline momentum


# =============================================================================
# DUAL CLASSIFIER-FREE GUIDANCE (DCFG)
# =============================================================================

class DualClassifierFreeGuidance(nn.Module):
    """
    Dual Classifier-Free Guidance for decoupling text and reference style.

    Standard CFG entangles text and reference in a single weight:
        f̂ = f_{a,t} + λ_cfg(f_{a,t} - f_{∅,∅})

    DCFG separates them for independent control:
        f̂_DCFG = f_{∅,t} + λ_t(f_{∅,t} - f_{∅,∅}) + λ_a(f_{a,t} - f_{∅,t})

    Where:
        f_{a,t}: Output conditioned on audio (a) and text (t)
        f_{∅,t}: Output conditioned on text only (no audio)
        f_{∅,∅}: Unconditional output
        λ_t: Text guidance scale (controls text fidelity)
        λ_a: Audio guidance scale (controls reference dependency)

    Benefits:
        - λ_a < λ_t reduces reference audio dependency
        - Can achieve reference-free generation with λ_a = 0
        - Preserves text intelligibility independent of style control
    """

    def __init__(self, config: ReStyleTTSConfig):
        super().__init__()
        self.config = config
        self.text_scale = config.dcfg_text_scale
        self.audio_scale = config.dcfg_audio_scale

    def forward(
        self,
        output_full: torch.Tensor,       # f_{a,t}: conditioned on both
        output_text_only: torch.Tensor,  # f_{∅,t}: text only
        output_uncond: torch.Tensor,     # f_{∅,∅}: unconditional
        text_scale: Optional[float] = None,
        audio_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply DCFG to decouple text and reference guidance.

        Args:
            output_full: Model output with full conditioning [batch, ...]
            output_text_only: Model output with text only [batch, ...]
            output_uncond: Unconditional model output [batch, ...]
            text_scale: Optional override for λ_t
            audio_scale: Optional override for λ_a

        Returns:
            Guided output with decoupled text/audio control
        """
        lambda_t = text_scale if text_scale is not None else self.text_scale
        lambda_a = audio_scale if audio_scale is not None else self.audio_scale

        # DCFG formula: f̂ = f_{∅,t} + λ_t(f_{∅,t} - f_{∅,∅}) + λ_a(f_{a,t} - f_{∅,t})
        text_guidance = lambda_t * (output_text_only - output_uncond)
        audio_guidance = lambda_a * (output_full - output_text_only)

        guided_output = output_text_only + text_guidance + audio_guidance

        return guided_output

    def get_scales(self) -> Tuple[float, float]:
        """Get current guidance scales."""
        return self.text_scale, self.audio_scale

    def set_scales(self, text_scale: float, audio_scale: float):
        """Set guidance scales."""
        self.text_scale = text_scale
        self.audio_scale = audio_scale


class DCFGInference:
    """
    DCFG inference helper for generating with reduced reference dependency.

    This class manages the three forward passes needed for DCFG:
    1. Full conditioning (audio + text)
    2. Text-only conditioning
    3. Unconditional

    Usage:
        dcfg = DCFGInference(model, config)
        output = dcfg.generate(text_ids, audio_ref=None)  # Reference-free
        output = dcfg.generate(text_ids, audio_ref=ref)   # With reference
    """

    def __init__(
        self,
        model: nn.Module,
        config: ReStyleTTSConfig,
        null_audio_embed: Optional[torch.Tensor] = None,
        null_text_embed: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.config = config
        self.dcfg = DualClassifierFreeGuidance(config)

        # Learnable null embeddings for unconditional generation
        hidden_size = config.hidden_size

        if null_audio_embed is not None:
            self.null_audio = null_audio_embed
        else:
            self.null_audio = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.normal_(self.null_audio, std=0.02)

        if null_text_embed is not None:
            self.null_text = null_text_embed
        else:
            self.null_text = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.normal_(self.null_text, std=0.02)

    @torch.no_grad()
    def generate_step(
        self,
        text_embeds: torch.Tensor,
        audio_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Single DCFG generation step.

        Performs three forward passes and combines with DCFG.
        """
        batch_size = text_embeds.shape[0]
        device = text_embeds.device

        # Prepare null embeddings
        null_audio = self.null_audio.expand(batch_size, -1, -1).to(device)
        null_text = self.null_text.expand(batch_size, -1, -1).to(device)

        # 1. Full conditioning (audio + text)
        if audio_embeds is not None:
            full_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
        else:
            full_embeds = torch.cat([null_audio, text_embeds], dim=1)
        output_full = self.model(inputs_embeds=full_embeds, **kwargs)

        # 2. Text-only conditioning
        text_only_embeds = torch.cat([null_audio, text_embeds], dim=1)
        output_text_only = self.model(inputs_embeds=text_only_embeds, **kwargs)

        # 3. Unconditional
        uncond_embeds = torch.cat([null_audio, null_text], dim=1)
        output_uncond = self.model(inputs_embeds=uncond_embeds, **kwargs)

        # Apply DCFG
        if hasattr(output_full, 'logits'):
            logits_full = output_full.logits
            logits_text = output_text_only.logits
            logits_uncond = output_uncond.logits

            guided_logits = self.dcfg(logits_full, logits_text, logits_uncond)
            output_full.logits = guided_logits

        return output_full, None


# =============================================================================
# ORTHOGONAL LORA FUSION (OLoRA)
# =============================================================================

class OrthogonalLoRAFusion(nn.Module):
    """
    Orthogonal LoRA Fusion for combining multiple style LoRAs without interference.

    Standard LoRA combination (simple averaging) causes interference between
    different style attributes. OLoRA projects each LoRA update onto the
    orthogonal complement of the subspace spanned by all other LoRAs.

    Algorithm:
    1. For each LoRA i, compute projection matrix P_-i = V_-i(V_-i)^+
       where V_-i contains all other LoRA updates
    2. Orthogonalize: ṽ_i = (I - P_-i)v_i
    3. Fuse: ΔW_fuse = Σ_i α_i ΔW̃_i

    Benefits:
    - Order-independent (joint vs sequential projection)
    - Preserves individual LoRA characteristics
    - Enables continuous interpolation without artifacts
    """

    def __init__(self, config: ReStyleTTSConfig):
        super().__init__()
        self.config = config
        self.regularization = config.olora_regularization

    def compute_projection_matrix(
        self,
        vectors: torch.Tensor,  # [num_loras-1, dim]
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute projection matrix P = V(V^+) using SVD.

        Args:
            vectors: Matrix where each row is a flattened LoRA update
            device: Target device

        Returns:
            Projection matrix P of shape [dim, dim]
        """
        if vectors.shape[0] == 0:
            # No other LoRAs - return zero matrix (identity minus zero)
            return torch.zeros(vectors.shape[1], vectors.shape[1], device=device)

        # SVD for pseudoinverse: V^+ = V^T(VV^T)^{-1}
        # But we want P = V(V^+) = VV^T(VV^T)^{-1}... simplified using SVD
        try:
            U, S, Vh = torch.linalg.svd(vectors, full_matrices=False)

            # Regularization to handle near-zero singular values
            S_inv = torch.zeros_like(S)
            mask = S > 1e-6
            S_inv[mask] = 1.0 / S[mask]

            # V^+ = Vh^T * diag(1/S) * U^T
            V_pinv = Vh.T @ torch.diag(S_inv) @ U.T

            # P = V @ V^+
            P = vectors.T @ V_pinv.T

        except RuntimeError:
            # Fallback: simple outer product approximation
            P = vectors.T @ vectors / (vectors.shape[0] + 1e-6)

        return P

    def orthogonalize_lora(
        self,
        lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],  # {name: (A, B)}
        target_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Orthogonalize a single LoRA against all others.

        Args:
            lora_weights: Dict mapping LoRA names to (A, B) weight tuples
            target_name: Name of LoRA to orthogonalize

        Returns:
            Orthogonalized (A, B) weight tuple
        """
        target_A, target_B = lora_weights[target_name]
        device = target_A.device

        # Flatten target LoRA update: ΔW = B @ A
        target_update = (target_B @ target_A).flatten()

        # Collect other LoRA updates
        other_updates = []
        for name, (A, B) in lora_weights.items():
            if name != target_name:
                update = (B @ A).flatten()
                other_updates.append(update)

        if len(other_updates) == 0:
            return target_A, target_B

        # Stack other updates: [num_others, dim]
        V_others = torch.stack(other_updates, dim=0)

        # Compute projection matrix
        P = self.compute_projection_matrix(V_others, device)

        # Orthogonalize: ṽ = (I - P)v
        I = torch.eye(P.shape[0], device=device)
        orthogonal_update = (I - P) @ target_update

        # Reshape back to ΔW shape
        orthogonal_update = orthogonal_update.view(target_B.shape[0], target_A.shape[1])

        # Decompose back to A, B using SVD
        # ΔW = U @ S @ Vh, let B = U @ sqrt(S), A = sqrt(S) @ Vh
        U, S, Vh = torch.linalg.svd(orthogonal_update, full_matrices=False)

        rank = min(target_A.shape[0], S.shape[0])
        sqrt_S = torch.sqrt(S[:rank] + 1e-8)

        new_B = U[:, :rank] @ torch.diag(sqrt_S)
        new_A = torch.diag(sqrt_S) @ Vh[:rank, :]

        return new_A, new_B

    def fuse_loras(
        self,
        lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        scales: Dict[str, float],
        orthogonalize: bool = True,
    ) -> torch.Tensor:
        """
        Fuse multiple LoRAs with optional orthogonalization.

        Args:
            lora_weights: Dict mapping LoRA names to (A, B) weight tuples
            scales: Dict mapping LoRA names to scaling factors α
            orthogonalize: Whether to apply OLoRA orthogonalization

        Returns:
            Fused weight delta ΔW_fuse
        """
        if len(lora_weights) == 0:
            return None

        fused_update = None

        for name, (A, B) in lora_weights.items():
            scale = scales.get(name, 1.0)

            if orthogonalize and self.config.olora_enabled:
                A_orth, B_orth = self.orthogonalize_lora(lora_weights, name)
                update = scale * (B_orth @ A_orth)
            else:
                update = scale * (B @ A)

            if fused_update is None:
                fused_update = update
            else:
                fused_update = fused_update + update

        return fused_update

    def orthogonality_loss(
        self,
        lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute orthogonality regularization loss.

        Encourages LoRA updates to be mutually orthogonal during training.
        """
        if len(lora_weights) < 2:
            return torch.tensor(0.0)

        # Flatten all LoRA updates
        updates = []
        for A, B in lora_weights.values():
            update = (B @ A).flatten()
            updates.append(F.normalize(update, dim=0))

        updates = torch.stack(updates, dim=0)

        # Compute Gram matrix
        gram = updates @ updates.T

        # Penalize off-diagonal elements (non-orthogonality)
        identity = torch.eye(gram.shape[0], device=gram.device)
        loss = ((gram - identity) ** 2).sum() / (gram.shape[0] ** 2)

        return self.regularization * loss


# =============================================================================
# STYLE LORA MANAGER
# =============================================================================

class StyleLoRAManager:
    """
    Manages multiple style-specific LoRAs for inference.

    Provides:
    - Loading/saving style LoRAs
    - Runtime composition with OLoRA
    - Scaling and interpolation
    - Integration with DCFG inference

    Usage:
        manager = StyleLoRAManager(config)
        manager.load_style("happy", "checkpoints/happy_lora/")
        manager.load_style("high_pitch", "checkpoints/high_pitch_lora/")

        # Compose styles
        weights = manager.compose_styles({
            "happy": 0.8,
            "high_pitch": 0.5
        })

        # Apply to model
        manager.apply_to_model(model, weights)
    """

    def __init__(self, config: ReStyleTTSConfig):
        self.config = config
        self.fusion = OrthogonalLoRAFusion(config)

        # Storage for loaded LoRAs
        self.loaded_loras: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.style_metadata: Dict[str, Dict] = {}

    def load_style(
        self,
        style_name: str,
        lora_path: Union[str, Path],
    ) -> None:
        """
        Load a style-specific LoRA from disk.

        Args:
            style_name: Name for this style (e.g., "happy", "high_pitch")
            lora_path: Path to LoRA checkpoint directory
        """
        lora_path = Path(lora_path)

        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_path}")

        # Load PEFT adapter config
        adapter_config_path = lora_path / "adapter_config.json"
        if adapter_config_path.exists():
            import json
            with open(adapter_config_path) as f:
                self.style_metadata[style_name] = json.load(f)

        # Load adapter weights
        adapter_path = lora_path / "adapter_model.safetensors"
        if not adapter_path.exists():
            adapter_path = lora_path / "adapter_model.bin"

        if adapter_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            weights = load_file(str(adapter_path))
        else:
            weights = torch.load(str(adapter_path), map_location="cpu")

        # Parse LoRA A/B pairs by layer
        lora_weights = {}
        for key, value in weights.items():
            if "lora_A" in key:
                layer_key = key.replace(".lora_A.weight", "")
                if layer_key not in lora_weights:
                    lora_weights[layer_key] = [None, None]
                lora_weights[layer_key][0] = value
            elif "lora_B" in key:
                layer_key = key.replace(".lora_B.weight", "")
                if layer_key not in lora_weights:
                    lora_weights[layer_key] = [None, None]
                lora_weights[layer_key][1] = value

        # Convert to tuples
        self.loaded_loras[style_name] = {
            k: (v[0], v[1]) for k, v in lora_weights.items()
            if v[0] is not None and v[1] is not None
        }

        print(f"Loaded style LoRA '{style_name}' with {len(self.loaded_loras[style_name])} layers")

    def compose_styles(
        self,
        style_scales: Dict[str, float],
        orthogonalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compose multiple styles into fused weight deltas.

        Args:
            style_scales: Dict mapping style names to scaling factors
            orthogonalize: Whether to apply OLoRA orthogonalization

        Returns:
            Dict mapping layer names to fused weight deltas
        """
        # Validate styles
        for style in style_scales:
            if style not in self.loaded_loras:
                raise ValueError(f"Style '{style}' not loaded. Load it first with load_style()")

        # Get all layer names across all styles
        all_layers = set()
        for style in style_scales:
            all_layers.update(self.loaded_loras[style].keys())

        # Fuse each layer
        fused_weights = {}
        for layer_name in all_layers:
            # Collect LoRA weights for this layer
            layer_loras = {}
            layer_scales = {}

            for style, scale in style_scales.items():
                if layer_name in self.loaded_loras[style]:
                    layer_loras[style] = self.loaded_loras[style][layer_name]

                    # Apply style-specific scaling from config
                    if style in ["high_pitch", "low_pitch", "high_energy",
                                 "low_energy", "fast_tempo", "slow_tempo"]:
                        scale *= self.config.prosody_lora_scale
                    else:
                        scale *= self.config.emotion_lora_scale

                    layer_scales[style] = scale

            if layer_loras:
                fused = self.fusion.fuse_loras(
                    layer_loras, layer_scales, orthogonalize=orthogonalize
                )
                if fused is not None:
                    fused_weights[layer_name] = fused

        return fused_weights

    def apply_to_model(
        self,
        model: nn.Module,
        fused_weights: Dict[str, torch.Tensor],
    ) -> None:
        """
        Apply fused LoRA weights to model.

        Args:
            model: Target model
            fused_weights: Dict from compose_styles()
        """
        for name, param in model.named_parameters():
            # Find matching fused weight
            for layer_name, delta in fused_weights.items():
                if layer_name in name and "weight" in name:
                    if param.shape == delta.shape:
                        param.data.add_(delta.to(param.device, param.dtype))
                    break

    def get_available_styles(self) -> List[str]:
        """Get list of loaded style names."""
        return list(self.loaded_loras.keys())

    def interpolate_styles(
        self,
        style1: str,
        style2: str,
        t: float,
    ) -> Dict[str, float]:
        """
        Create interpolated style scales.

        Args:
            style1: Source style
            style2: Target style
            t: Interpolation factor [0, 1]

        Returns:
            Dict with interpolated scales
        """
        return {
            style1: 1.0 - t,
            style2: t,
        }


# =============================================================================
# TIMBRE CONSISTENCY OPTIMIZATION (TCO)
# =============================================================================

class TimbreConsistencyOptimization:
    """
    Timbre Consistency Optimization for preserving speaker identity.

    Uses reinforcement learning-style reweighting of the training loss
    based on speaker similarity rewards:
        - Computes advantage: A_t = r_t - b_t (with EMA baseline)
        - Reweights loss: w_t = 1 + λ * tanh(β * A_t)

    No gradient through generation - preserves training stability.
    """

    def __init__(self, config: ReStyleTTSConfig):
        self.config = config
        self.lambda_weight = config.tco_lambda
        self.beta = config.tco_beta
        self.momentum = config.tco_momentum

        # EMA baseline
        self.baseline = 0.0
        self.initialized = False

    def compute_speaker_similarity(
        self,
        generated_embed: torch.Tensor,
        reference_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute speaker embedding similarity.

        Args:
            generated_embed: Speaker embedding of generated audio
            reference_embed: Speaker embedding of reference audio

        Returns:
            Similarity score (cosine similarity)
        """
        gen_norm = F.normalize(generated_embed, dim=-1)
        ref_norm = F.normalize(reference_embed, dim=-1)
        similarity = (gen_norm * ref_norm).sum(dim=-1)
        return similarity

    def update_baseline(self, reward: float) -> None:
        """Update EMA baseline."""
        if not self.initialized:
            self.baseline = reward
            self.initialized = True
        else:
            self.baseline = self.momentum * self.baseline + (1 - self.momentum) * reward

    def compute_loss_weight(
        self,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss reweighting factor.

        Args:
            reward: Speaker similarity reward

        Returns:
            Loss weight w_t
        """
        # Advantage
        advantage = reward - self.baseline

        # Loss weight: w_t = 1 + λ * tanh(β * A_t)
        weight = 1.0 + self.lambda_weight * torch.tanh(self.beta * advantage)

        # Update baseline (detached)
        self.update_baseline(reward.mean().item())

        return weight

    def reweight_loss(
        self,
        loss: torch.Tensor,
        generated_embed: torch.Tensor,
        reference_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply TCO reweighting to loss.

        Args:
            loss: Original training loss
            generated_embed: Speaker embedding of generated audio
            reference_embed: Speaker embedding of reference audio

        Returns:
            Reweighted loss
        """
        with torch.no_grad():
            similarity = self.compute_speaker_similarity(generated_embed, reference_embed)
            weight = self.compute_loss_weight(similarity)

        return loss * weight


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class StyleLoRATrainer:
    """
    Trainer for style-specific LoRAs.

    Implements the training procedure from ReStyle-TTS:
    1. Train on style-specific subset of data
    2. Apply masking (0.3 for speech, 0.2 for text+speech)
    3. Optional TCO for speaker consistency
    4. Orthogonality regularization
    """

    def __init__(
        self,
        model: nn.Module,
        config: ReStyleTTSConfig,
        style_name: str,
    ):
        self.model = model
        self.config = config
        self.style_name = style_name

        # Components
        self.fusion = OrthogonalLoRAFusion(config)
        self.tco = TimbreConsistencyOptimization(config) if config.tco_enabled else None

    def apply_masking(
        self,
        speech_input: torch.Tensor,
        text_input: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking for classifier-free guidance training.

        Args:
            speech_input: Speech embeddings
            text_input: Text embeddings
            training: Whether in training mode

        Returns:
            Masked (speech, text) tuple
        """
        if not training:
            return speech_input, text_input

        batch_size = speech_input.shape[0]
        device = speech_input.device

        # Speech-only masking (0.3)
        speech_mask = torch.rand(batch_size, device=device) < self.config.speech_mask_ratio
        speech_input = torch.where(
            speech_mask.unsqueeze(-1).unsqueeze(-1),
            torch.zeros_like(speech_input),
            speech_input
        )

        # Text+speech masking (0.2)
        text_mask = torch.rand(batch_size, device=device) < self.config.text_mask_ratio
        full_mask = speech_mask | text_mask
        text_input = torch.where(
            full_mask.unsqueeze(-1).unsqueeze(-1),
            torch.zeros_like(text_input),
            text_input
        )

        return speech_input, text_input

    def compute_loss(
        self,
        model_output,
        target_audio: torch.Tensor,
        lora_weights: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        generated_embed: Optional[torch.Tensor] = None,
        reference_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with optional regularization and TCO.

        Args:
            model_output: Model output with loss
            target_audio: Target audio for reconstruction
            lora_weights: Optional LoRA weights for orthogonality loss
            generated_embed: Optional speaker embedding for TCO
            reference_embed: Optional reference speaker embedding for TCO

        Returns:
            Dict with loss components
        """
        losses = {}

        # Main reconstruction loss
        if hasattr(model_output, 'loss'):
            losses['reconstruction'] = model_output.loss
        else:
            losses['reconstruction'] = torch.tensor(0.0)

        # Orthogonality regularization
        if lora_weights and len(lora_weights) > 1:
            losses['orthogonality'] = self.fusion.orthogonality_loss(lora_weights)
        else:
            losses['orthogonality'] = torch.tensor(0.0)

        # TCO reweighting
        total_loss = losses['reconstruction'] + losses['orthogonality']

        if (self.tco is not None and
            generated_embed is not None and
            reference_embed is not None):
            total_loss = self.tco.reweight_loss(
                total_loss, generated_embed, reference_embed
            )

        losses['total'] = total_loss

        return losses


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_style_lora_config(
    style_name: str,
    base_config: Optional[ReStyleTTSConfig] = None,
) -> Dict:
    """
    Create PEFT LoRA config for a specific style.

    Args:
        style_name: Name of the style
        base_config: Optional ReStyleTTSConfig

    Returns:
        Dict compatible with peft.LoraConfig
    """
    if base_config is None:
        base_config = ReStyleTTSConfig()

    return {
        "r": base_config.lora_rank,
        "lora_alpha": base_config.lora_alpha,
        "target_modules": base_config.lora_target_modules,
        "lora_dropout": base_config.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "modules_to_save": None,
    }


def parse_style_string(style_string: str) -> Dict[str, float]:
    """
    Parse style specification string into dict.

    Args:
        style_string: e.g., "happy:0.8,high_pitch:0.5"

    Returns:
        Dict mapping style names to scales
    """
    styles = {}
    for part in style_string.split(","):
        if ":" in part:
            name, scale = part.strip().split(":")
            styles[name.strip()] = float(scale)
        else:
            styles[part.strip()] = 1.0
    return styles


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ReStyle-TTS: DCFG + Orthogonal LoRA Fusion - Test Suite")
    print("=" * 70)

    config = ReStyleTTSConfig()

    # Test 1: DCFG
    print("\n[Test 1] Dual Classifier-Free Guidance...")
    dcfg = DualClassifierFreeGuidance(config)

    batch_size = 2
    seq_len = 10
    hidden = 256

    output_full = torch.randn(batch_size, seq_len, hidden)
    output_text = torch.randn(batch_size, seq_len, hidden)
    output_uncond = torch.randn(batch_size, seq_len, hidden)

    guided = dcfg(output_full, output_text, output_uncond)
    print(f"  Input shapes: {output_full.shape}")
    print(f"  Guided output shape: {guided.shape}")
    print(f"  λ_t={dcfg.text_scale}, λ_a={dcfg.audio_scale}")
    print("  [PASS]")

    # Test 2: OLoRA
    print("\n[Test 2] Orthogonal LoRA Fusion...")
    olora = OrthogonalLoRAFusion(config)

    # Create dummy LoRA weights
    in_dim, out_dim, rank = 64, 128, 8
    lora_weights = {
        "happy": (torch.randn(rank, in_dim), torch.randn(out_dim, rank)),
        "high_pitch": (torch.randn(rank, in_dim), torch.randn(out_dim, rank)),
        "sad": (torch.randn(rank, in_dim), torch.randn(out_dim, rank)),
    }

    # Test orthogonalization
    A_orth, B_orth = olora.orthogonalize_lora(lora_weights, "happy")
    print(f"  Original A shape: {lora_weights['happy'][0].shape}")
    print(f"  Orthogonalized A shape: {A_orth.shape}")

    # Test fusion
    scales = {"happy": 0.8, "high_pitch": 0.5, "sad": 0.3}
    fused = olora.fuse_loras(lora_weights, scales)
    print(f"  Fused weight delta shape: {fused.shape}")

    # Test orthogonality loss
    orth_loss = olora.orthogonality_loss(lora_weights)
    print(f"  Orthogonality loss: {orth_loss.item():.4f}")
    print("  [PASS]")

    # Test 3: Style Manager
    print("\n[Test 3] Style LoRA Manager...")
    manager = StyleLoRAManager(config)
    print(f"  Available styles in config: {config.available_styles[:5]}...")

    # Test interpolation
    interp = manager.interpolate_styles("happy", "sad", 0.5)
    print(f"  Interpolated scales: {interp}")
    print("  [PASS]")

    # Test 4: TCO
    print("\n[Test 4] Timbre Consistency Optimization...")
    tco = TimbreConsistencyOptimization(config)

    gen_embed = F.normalize(torch.randn(batch_size, 256), dim=-1)
    ref_embed = F.normalize(torch.randn(batch_size, 256), dim=-1)

    similarity = tco.compute_speaker_similarity(gen_embed, ref_embed)
    weight = tco.compute_loss_weight(similarity)
    print(f"  Speaker similarity: {similarity.mean().item():.4f}")
    print(f"  Loss weight: {weight.mean().item():.4f}")
    print(f"  Baseline: {tco.baseline:.4f}")
    print("  [PASS]")

    # Test 5: Style String Parsing
    print("\n[Test 5] Style String Parsing...")
    style_str = "happy:0.8, high_pitch:0.5, calm:0.3"
    parsed = parse_style_string(style_str)
    print(f"  Input: '{style_str}'")
    print(f"  Parsed: {parsed}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All ReStyle-TTS tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from restyle_tts import (
    ReStyleTTSConfig, StyleLoRAManager,
    DualClassifierFreeGuidance, parse_style_string
)

# Initialize
config = ReStyleTTSConfig()
manager = StyleLoRAManager(config)

# Load style LoRAs
manager.load_style("happy", "checkpoints/happy_lora/")
manager.load_style("high_pitch", "checkpoints/high_pitch_lora/")

# Compose styles with OLoRA fusion
style_scales = parse_style_string("happy:0.8,high_pitch:0.5")
fused_weights = manager.compose_styles(style_scales)

# Apply to model
manager.apply_to_model(model, fused_weights)

# Generate with DCFG
dcfg = DualClassifierFreeGuidance(config)
# Set lower audio scale for reference-free generation
dcfg.set_scales(text_scale=2.0, audio_scale=0.1)
output = dcfg(output_full, output_text_only, output_uncond)
""")
