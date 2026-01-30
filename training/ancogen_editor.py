"""
AnCoGen: Attribute-Controllable Neural Codec Generation

Based on "Bringing Interpretability to Neural Audio Codecs" (Interspeech 2025)
arXiv:2506.04492

Key Technique: AnCoGen models for attribute extraction and editing in codec space.
Enables direct manipulation of speech attributes (pitch, speaker, emotion) at
the discrete token level without regenerating from scratch.

Key Findings from Paper:
- Content dominates early RVQ scales (layers 1-2)
- Speaker identity emerges in later stages (layers 3-4+)
- Pitch remains poorly disentangled in standard codecs
- Direct attribute manipulation possible via:
  1. Attribute extraction networks per layer
  2. Attribute injection via residual addition or concatenation
  3. Layer-specific editing (early for content, late for speaker)

Architecture:
```
                    ┌─────────────────────────────────────────┐
                    │         AnCoGen Editor Pipeline          │
                    │                                         │
 Input Codes ──────►│  [Layer Extractor] ──► Attribute Embed  │
                    │         ↓                               │
                    │  [Attribute Modifier] ◄── Target Attr   │
                    │         ↓                               │
                    │  [Injection Network] ──► Modified Codes │
                    │                                         │
                    └─────────────────────────────────────────┘
```

Benefits for our pipeline:
- Enable post-hoc prosody editing without re-synthesis
- Transfer prosody from one utterance to another
- Fine-grained attribute control at specific RVQ layers
- Debug and improve codec disentanglement

Usage:
    from training.ancogen_editor import (
        AnCoGenEditor,
        AnCoGenConfig,
        AttributeExtractor,
        AttributeInjector,
        layer_aware_attribute_edit,
    )

    # Initialize editor
    editor = AnCoGenEditor(config)

    # Extract attributes from codes
    attributes = editor.extract_attributes(rvq_codes, rvq_embeddings)
    print(f"Speaker embedding: {attributes['speaker'].shape}")
    print(f"Pitch embedding: {attributes['pitch'].shape}")

    # Modify specific attribute
    modified_codes = editor.edit_attribute(
        rvq_codes,
        rvq_embeddings,
        attribute='speaker',
        target_value=target_speaker_emb,
        layers=[3, 4, 5, 6, 7],  # Later layers for speaker
    )

    # Prosody transfer
    transferred = editor.transfer_prosody(
        content_codes=codes_a,       # Content from A
        prosody_codes=codes_b,       # Prosody from B
    )
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AnCoGenConfig:
    """Configuration for AnCoGen attribute editor."""

    # RVQ structure
    num_rvq_layers: int = 8              # Number of RVQ quantizer layers
    codebook_size: int = 1024            # Size of each codebook
    embedding_dim: int = 256             # Dimension of code embeddings

    # Attribute dimensions
    speaker_dim: int = 256               # Speaker embedding dimension
    pitch_dim: int = 64                  # Pitch embedding dimension
    content_dim: int = 256               # Content embedding dimension
    emotion_dim: int = 64                # Emotion embedding dimension
    energy_dim: int = 32                 # Energy embedding dimension

    # Layer assignments (based on AnCoGen findings)
    content_layers: List[int] = field(default_factory=lambda: [0, 1])
    speaker_layers: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])
    pitch_layers: List[int] = field(default_factory=lambda: [2, 3, 4])
    energy_layers: List[int] = field(default_factory=lambda: [1, 2])
    emotion_layers: List[int] = field(default_factory=lambda: [2, 3, 4, 5])

    # Network architecture
    extractor_hidden_dim: int = 512      # Hidden dimension for extractors
    extractor_num_layers: int = 2        # Number of hidden layers
    injector_hidden_dim: int = 512       # Hidden dimension for injectors
    injector_num_layers: int = 2

    # Training
    dropout: float = 0.1
    use_residual_injection: bool = True  # Add modified vs replace
    use_layer_norm: bool = True

    # Editing parameters
    edit_strength: float = 1.0           # How strongly to apply edits
    blend_mode: str = "residual"         # residual, replace, or interpolate


# =============================================================================
# ATTRIBUTE EXTRACTOR
# =============================================================================

class AttributeExtractor(nn.Module):
    """
    Extracts speech attributes from RVQ layer embeddings.

    Each attribute (speaker, pitch, content, etc.) is extracted from
    specific layers identified through interpretability analysis.
    """

    def __init__(
        self,
        config: AnCoGenConfig,
        attribute_name: str,
        attribute_dim: int,
        source_layers: List[int],
    ):
        super().__init__()
        self.config = config
        self.attribute_name = attribute_name
        self.attribute_dim = attribute_dim
        self.source_layers = source_layers

        # Input: concatenated embeddings from source layers
        input_dim = config.embedding_dim * len(source_layers)

        # Build extraction network
        layers = []
        prev_dim = input_dim

        for i in range(config.extractor_num_layers):
            layers.extend([
                nn.Linear(prev_dim, config.extractor_hidden_dim),
                nn.LayerNorm(config.extractor_hidden_dim) if config.use_layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            prev_dim = config.extractor_hidden_dim

        # Output projection
        layers.append(nn.Linear(prev_dim, attribute_dim))

        self.network = nn.Sequential(*layers)

        # Optional: attention over layers
        self.layer_attention = nn.Sequential(
            nn.Linear(config.embedding_dim, 1),
            nn.Softmax(dim=-2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        layer_embeddings: Dict[int, torch.Tensor],  # layer -> [batch, seq, dim]
        pool_sequence: bool = True,
    ) -> torch.Tensor:
        """
        Extract attribute from layer embeddings.

        Args:
            layer_embeddings: Dictionary mapping layer index to embeddings
            pool_sequence: Whether to pool over sequence dimension

        Returns:
            [batch, attribute_dim] or [batch, seq, attribute_dim]
        """
        # Gather embeddings from source layers
        source_embs = []
        for layer_idx in self.source_layers:
            if layer_idx in layer_embeddings:
                source_embs.append(layer_embeddings[layer_idx])

        if not source_embs:
            raise ValueError(f"No embeddings found for layers {self.source_layers}")

        # Stack and compute attention weights
        stacked = torch.stack(source_embs, dim=-2)  # [batch, seq, num_layers, dim]

        # Attention-weighted combination
        attn_weights = self.layer_attention(stacked)  # [batch, seq, num_layers, 1]
        weighted = (stacked * attn_weights).sum(dim=-2)  # [batch, seq, dim]

        # Also concatenate for network input
        batch_size, seq_len = source_embs[0].shape[:2]
        concat = torch.cat(source_embs, dim=-1)  # [batch, seq, dim * num_layers]

        # Extract attribute
        if pool_sequence:
            concat = concat.mean(dim=1)  # [batch, dim * num_layers]

        attribute = self.network(concat)

        return attribute


class MultiAttributeExtractor(nn.Module):
    """
    Extracts multiple attributes from RVQ embeddings.
    """

    def __init__(self, config: AnCoGenConfig):
        super().__init__()
        self.config = config

        # Create extractors for each attribute
        self.extractors = nn.ModuleDict({
            'speaker': AttributeExtractor(
                config, 'speaker', config.speaker_dim, config.speaker_layers
            ),
            'pitch': AttributeExtractor(
                config, 'pitch', config.pitch_dim, config.pitch_layers
            ),
            'content': AttributeExtractor(
                config, 'content', config.content_dim, config.content_layers
            ),
            'emotion': AttributeExtractor(
                config, 'emotion', config.emotion_dim, config.emotion_layers
            ),
            'energy': AttributeExtractor(
                config, 'energy', config.energy_dim, config.energy_layers
            ),
        })

    def forward(
        self,
        layer_embeddings: Dict[int, torch.Tensor],
        pool_sequence: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all attributes.

        Returns:
            Dictionary mapping attribute name to embedding
        """
        return {
            name: extractor(layer_embeddings, pool_sequence)
            for name, extractor in self.extractors.items()
        }


# =============================================================================
# ATTRIBUTE INJECTOR
# =============================================================================

class AttributeInjector(nn.Module):
    """
    Injects modified attributes back into RVQ embeddings.

    Supports multiple injection modes:
    - residual: Add modification as residual
    - replace: Replace embedding entirely
    - interpolate: Blend original and modified
    """

    def __init__(
        self,
        config: AnCoGenConfig,
        attribute_name: str,
        attribute_dim: int,
        target_layers: List[int],
    ):
        super().__init__()
        self.config = config
        self.attribute_name = attribute_name
        self.attribute_dim = attribute_dim
        self.target_layers = target_layers

        # Build injection network per layer
        self.layer_networks = nn.ModuleDict()

        for layer_idx in target_layers:
            self.layer_networks[str(layer_idx)] = nn.Sequential(
                nn.Linear(attribute_dim, config.injector_hidden_dim),
                nn.LayerNorm(config.injector_hidden_dim) if config.use_layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.injector_hidden_dim, config.embedding_dim),
            )

        # Gating mechanism for residual injection
        if config.use_residual_injection:
            self.gates = nn.ModuleDict()
            for layer_idx in target_layers:
                self.gates[str(layer_idx)] = nn.Sequential(
                    nn.Linear(config.embedding_dim * 2, config.embedding_dim),
                    nn.Sigmoid(),
                )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        layer_embeddings: Dict[int, torch.Tensor],  # layer -> [batch, seq, dim]
        attribute: torch.Tensor,                     # [batch, attribute_dim]
        strength: float = 1.0,
    ) -> Dict[int, torch.Tensor]:
        """
        Inject attribute into layer embeddings.

        Args:
            layer_embeddings: Original embeddings per layer
            attribute: Attribute embedding to inject
            strength: Injection strength (0-1)

        Returns:
            Modified embeddings per layer
        """
        modified = {}

        for layer_idx, emb in layer_embeddings.items():
            if layer_idx not in self.target_layers:
                modified[layer_idx] = emb
                continue

            batch_size, seq_len, dim = emb.shape

            # Expand attribute to sequence length
            attr_expanded = attribute.unsqueeze(1).expand(-1, seq_len, -1)

            # Transform attribute to embedding space
            layer_key = str(layer_idx)
            attr_transformed = self.layer_networks[layer_key](attr_expanded)

            if self.config.blend_mode == "replace":
                modified[layer_idx] = attr_transformed * strength + emb * (1 - strength)

            elif self.config.blend_mode == "residual":
                if self.config.use_residual_injection:
                    # Gated residual
                    concat = torch.cat([emb, attr_transformed], dim=-1)
                    gate = self.gates[layer_key](concat)
                    residual = gate * attr_transformed
                else:
                    residual = attr_transformed

                modified[layer_idx] = emb + strength * residual

            elif self.config.blend_mode == "interpolate":
                modified[layer_idx] = (1 - strength) * emb + strength * attr_transformed

            else:
                modified[layer_idx] = emb

        return modified


class MultiAttributeInjector(nn.Module):
    """
    Injects multiple attributes into RVQ embeddings.
    """

    def __init__(self, config: AnCoGenConfig):
        super().__init__()
        self.config = config

        # Create injectors for each attribute
        self.injectors = nn.ModuleDict({
            'speaker': AttributeInjector(
                config, 'speaker', config.speaker_dim, config.speaker_layers
            ),
            'pitch': AttributeInjector(
                config, 'pitch', config.pitch_dim, config.pitch_layers
            ),
            'content': AttributeInjector(
                config, 'content', config.content_dim, config.content_layers
            ),
            'emotion': AttributeInjector(
                config, 'emotion', config.emotion_dim, config.emotion_layers
            ),
            'energy': AttributeInjector(
                config, 'energy', config.energy_dim, config.energy_layers
            ),
        })

    def forward(
        self,
        layer_embeddings: Dict[int, torch.Tensor],
        attributes: Dict[str, torch.Tensor],
        strengths: Optional[Dict[str, float]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Inject multiple attributes sequentially.

        Args:
            layer_embeddings: Original embeddings
            attributes: Attributes to inject
            strengths: Per-attribute injection strength

        Returns:
            Modified embeddings
        """
        strengths = strengths or {}
        modified = layer_embeddings

        for attr_name, attr_value in attributes.items():
            if attr_name in self.injectors:
                strength = strengths.get(attr_name, self.config.edit_strength)
                modified = self.injectors[attr_name](modified, attr_value, strength)

        return modified


# =============================================================================
# ANCOGEN EDITOR
# =============================================================================

class AnCoGenEditor(nn.Module):
    """
    Main AnCoGen editor for attribute-controllable codec generation.

    Combines extraction and injection networks for complete attribute editing.
    """

    def __init__(self, config: Optional[AnCoGenConfig] = None):
        super().__init__()
        self.config = config or AnCoGenConfig()

        # Attribute extraction
        self.extractor = MultiAttributeExtractor(self.config)

        # Attribute injection
        self.injector = MultiAttributeInjector(self.config)

        # Optional: codebook embedding lookup
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(self.config.codebook_size, self.config.embedding_dim)
            for _ in range(self.config.num_rvq_layers)
        ])

    def codes_to_embeddings(
        self,
        codes: torch.Tensor,  # [batch, num_layers, seq]
    ) -> Dict[int, torch.Tensor]:
        """
        Convert discrete codes to continuous embeddings.

        Args:
            codes: [batch, num_layers, seq] discrete code indices

        Returns:
            Dictionary mapping layer index to [batch, seq, dim] embeddings
        """
        layer_embeddings = {}

        for layer_idx in range(codes.shape[1]):
            layer_codes = codes[:, layer_idx, :]  # [batch, seq]
            embeddings = self.codebook_embeddings[layer_idx](layer_codes)
            layer_embeddings[layer_idx] = embeddings

        return layer_embeddings

    def extract_attributes(
        self,
        codes: Optional[torch.Tensor] = None,
        layer_embeddings: Optional[Dict[int, torch.Tensor]] = None,
        pool_sequence: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all attributes from codes or embeddings.

        Args:
            codes: [batch, num_layers, seq] discrete codes (optional)
            layer_embeddings: Pre-computed embeddings (optional)
            pool_sequence: Whether to pool over sequence

        Returns:
            Dictionary of extracted attributes
        """
        if layer_embeddings is None:
            if codes is None:
                raise ValueError("Must provide either codes or layer_embeddings")
            layer_embeddings = self.codes_to_embeddings(codes)

        return self.extractor(layer_embeddings, pool_sequence)

    def edit_attribute(
        self,
        codes: Optional[torch.Tensor] = None,
        layer_embeddings: Optional[Dict[int, torch.Tensor]] = None,
        attribute: str = 'speaker',
        target_value: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
        strength: float = 1.0,
    ) -> Dict[int, torch.Tensor]:
        """
        Edit a specific attribute in the embeddings.

        Args:
            codes: Discrete codes (optional)
            layer_embeddings: Pre-computed embeddings (optional)
            attribute: Which attribute to edit
            target_value: Target attribute embedding (replace mode)
            delta: Attribute change (additive mode)
            strength: Edit strength

        Returns:
            Modified layer embeddings
        """
        if layer_embeddings is None:
            if codes is None:
                raise ValueError("Must provide either codes or layer_embeddings")
            layer_embeddings = self.codes_to_embeddings(codes)

        # Determine attribute value
        if target_value is not None:
            attr_value = target_value
        elif delta is not None:
            # Extract current and add delta
            current = self.extractor.extractors[attribute](layer_embeddings)
            attr_value = current + delta
        else:
            raise ValueError("Must provide either target_value or delta")

        # Inject modified attribute
        return self.injector.injectors[attribute](
            layer_embeddings, attr_value, strength
        )

    def edit_multiple_attributes(
        self,
        codes: Optional[torch.Tensor] = None,
        layer_embeddings: Optional[Dict[int, torch.Tensor]] = None,
        attributes: Dict[str, torch.Tensor] = None,
        strengths: Optional[Dict[str, float]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Edit multiple attributes simultaneously.

        Args:
            codes: Discrete codes
            layer_embeddings: Pre-computed embeddings
            attributes: Dictionary of attribute name -> target value
            strengths: Per-attribute strengths

        Returns:
            Modified embeddings
        """
        if layer_embeddings is None:
            if codes is None:
                raise ValueError("Must provide either codes or layer_embeddings")
            layer_embeddings = self.codes_to_embeddings(codes)

        return self.injector(layer_embeddings, attributes, strengths)

    def transfer_attribute(
        self,
        source_codes: torch.Tensor,
        target_codes: torch.Tensor,
        attribute: str,
        strength: float = 1.0,
    ) -> Dict[int, torch.Tensor]:
        """
        Transfer an attribute from source to target.

        Args:
            source_codes: Codes to extract attribute from
            target_codes: Codes to modify
            attribute: Which attribute to transfer
            strength: Transfer strength

        Returns:
            Target embeddings with transferred attribute
        """
        # Extract attribute from source
        source_emb = self.codes_to_embeddings(source_codes)
        source_attr = self.extractor.extractors[attribute](source_emb)

        # Inject into target
        target_emb = self.codes_to_embeddings(target_codes)
        return self.edit_attribute(
            layer_embeddings=target_emb,
            attribute=attribute,
            target_value=source_attr,
            strength=strength,
        )

    def transfer_prosody(
        self,
        content_codes: torch.Tensor,
        prosody_codes: torch.Tensor,
        prosody_strength: float = 1.0,
    ) -> Dict[int, torch.Tensor]:
        """
        Transfer prosody (pitch, energy, emotion) from one utterance to another.

        Keeps content/speaker from content_codes, takes prosody from prosody_codes.

        Args:
            content_codes: Codes providing content and speaker identity
            prosody_codes: Codes providing prosody features
            prosody_strength: How strongly to apply prosody transfer

        Returns:
            Combined embeddings
        """
        # Extract prosody from source
        prosody_emb = self.codes_to_embeddings(prosody_codes)
        pitch = self.extractor.extractors['pitch'](prosody_emb)
        energy = self.extractor.extractors['energy'](prosody_emb)
        emotion = self.extractor.extractors['emotion'](prosody_emb)

        # Start with content codes
        result = self.codes_to_embeddings(content_codes)

        # Inject prosody attributes
        prosody_attrs = {
            'pitch': pitch,
            'energy': energy,
            'emotion': emotion,
        }

        return self.injector(
            result,
            prosody_attrs,
            strengths={k: prosody_strength for k in prosody_attrs}
        )

    def forward(
        self,
        codes: torch.Tensor,
        edits: Optional[Dict[str, torch.Tensor]] = None,
        strengths: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass: extract attributes and optionally apply edits.

        Args:
            codes: [batch, num_layers, seq] discrete codes
            edits: Optional attribute edits to apply
            strengths: Per-attribute edit strengths

        Returns:
            Dictionary with extracted attributes and modified embeddings
        """
        layer_embeddings = self.codes_to_embeddings(codes)

        # Extract attributes
        attributes = self.extractor(layer_embeddings)

        # Apply edits if provided
        if edits:
            modified_embeddings = self.injector(
                layer_embeddings,
                edits,
                strengths
            )
        else:
            modified_embeddings = layer_embeddings

        return {
            'attributes': attributes,
            'original_embeddings': layer_embeddings,
            'modified_embeddings': modified_embeddings,
        }


# =============================================================================
# ANCOGEN LOSS
# =============================================================================

class AnCoGenLoss(nn.Module):
    """
    Training loss for AnCoGen editor.

    Includes:
    - Reconstruction loss (extract + inject should recover original)
    - Attribute prediction loss (extracted attributes should match labels)
    - Orthogonality loss (different attributes should be independent)
    """

    def __init__(
        self,
        config: AnCoGenConfig,
        reconstruction_weight: float = 1.0,
        prediction_weight: float = 1.0,
        orthogonality_weight: float = 0.1,
    ):
        super().__init__()
        self.config = config
        self.reconstruction_weight = reconstruction_weight
        self.prediction_weight = prediction_weight
        self.orthogonality_weight = orthogonality_weight

        # Attribute predictors for supervised training
        self.attribute_predictors = nn.ModuleDict()

    def add_attribute_predictor(
        self,
        attribute_name: str,
        attribute_dim: int,
        num_classes: Optional[int] = None,
    ):
        """
        Add a predictor for supervised attribute training.

        Args:
            attribute_name: Name of attribute
            attribute_dim: Dimension of extracted embedding
            num_classes: Number of classes (None for regression)
        """
        if num_classes:
            self.attribute_predictors[attribute_name] = nn.Sequential(
                nn.Linear(attribute_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            )
        else:
            self.attribute_predictors[attribute_name] = nn.Sequential(
                nn.Linear(attribute_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

    def reconstruction_loss(
        self,
        original: Dict[int, torch.Tensor],
        reconstructed: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute reconstruction loss between original and reconstructed embeddings.
        """
        loss = 0.0
        num_layers = 0

        for layer_idx in original:
            if layer_idx in reconstructed:
                loss = loss + F.mse_loss(
                    reconstructed[layer_idx],
                    original[layer_idx]
                )
                num_layers += 1

        return loss / max(num_layers, 1)

    def prediction_loss(
        self,
        attributes: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute prediction loss for supervised attributes.
        """
        losses = {}

        for attr_name, attr_emb in attributes.items():
            if attr_name in labels and attr_name in self.attribute_predictors:
                pred = self.attribute_predictors[attr_name](attr_emb)
                target = labels[attr_name]

                if target.dim() == 1 and pred.shape[-1] > 1:
                    # Classification
                    losses[attr_name] = F.cross_entropy(pred, target.long())
                else:
                    # Regression
                    if target.dim() == 1:
                        target = target.unsqueeze(-1)
                    losses[attr_name] = F.mse_loss(pred, target)

        return losses

    def orthogonality_loss(
        self,
        attributes: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encourage orthogonality between different attributes.
        """
        attr_names = list(attributes.keys())
        loss = 0.0
        count = 0

        for i, name1 in enumerate(attr_names):
            for name2 in attr_names[i + 1:]:
                attr1 = attributes[name1]
                attr2 = attributes[name2]

                # Normalize
                attr1_norm = F.normalize(attr1, dim=-1)
                attr2_norm = F.normalize(attr2, dim=-1)

                # Cosine similarity should be low
                similarity = (attr1_norm * attr2_norm).sum(dim=-1).abs().mean()
                loss = loss + similarity
                count += 1

        return loss / max(count, 1)

    def forward(
        self,
        editor: AnCoGenEditor,
        codes: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            editor: AnCoGenEditor model
            codes: Input codes
            labels: Optional attribute labels for supervised training

        Returns:
            Dictionary of losses
        """
        # Forward pass
        output = editor(codes)
        attributes = output['attributes']
        original = output['original_embeddings']

        # Reconstruction loss: extract + inject should recover
        # First inject extracted attributes back
        reconstructed = editor.injector(
            original,
            attributes,
            strengths={k: 1.0 for k in attributes}
        )

        recon_loss = self.reconstruction_loss(original, reconstructed)

        # Prediction loss (if labels provided)
        pred_losses = {}
        total_pred_loss = torch.tensor(0.0, device=codes.device)

        if labels:
            pred_losses = self.prediction_loss(attributes, labels)
            if pred_losses:
                total_pred_loss = sum(pred_losses.values()) / len(pred_losses)

        # Orthogonality loss
        ortho_loss = self.orthogonality_loss(attributes)

        # Total loss
        total = (
            self.reconstruction_weight * recon_loss +
            self.prediction_weight * total_pred_loss +
            self.orthogonality_weight * ortho_loss
        )

        return {
            'total': total,
            'reconstruction': recon_loss,
            'prediction': total_pred_loss,
            'orthogonality': ortho_loss,
            **{f'pred_{k}': v for k, v in pred_losses.items()},
        }


# =============================================================================
# ADAPTER FOR CSM INTEGRATION
# =============================================================================

class AnCoGenAdapter(nn.Module):
    """
    Adapter for integrating AnCoGen with CSM prosody pipeline.

    Enables:
    - Extract prosody from existing codecs (FACodec, EnCodec, etc.)
    - Edit prosody and convert back to CSM prefix tokens
    - Prosody transfer between utterances
    """

    def __init__(
        self,
        config: AnCoGenConfig,
        output_dim: int = 2048,
        num_prefix_tokens: int = 4,
    ):
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        self.num_prefix_tokens = num_prefix_tokens

        # AnCoGen editor
        self.editor = AnCoGenEditor(config)

        # Project combined attributes to prefix tokens
        total_attr_dim = (
            config.speaker_dim +
            config.pitch_dim +
            config.content_dim +
            config.emotion_dim +
            config.energy_dim
        )

        self.prefix_projection = nn.Sequential(
            nn.Linear(total_attr_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim * num_prefix_tokens),
        )

    def forward(
        self,
        codes: torch.Tensor,
        edits: Optional[Dict[str, torch.Tensor]] = None,
        strengths: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process codes and generate prefix tokens.

        Args:
            codes: [batch, num_layers, seq] discrete codes
            edits: Optional attribute edits
            strengths: Edit strengths

        Returns:
            Dictionary with prosody_tokens and attributes
        """
        # Get attributes (with optional edits)
        result = self.editor(codes, edits, strengths)
        attributes = result['attributes']

        # Concatenate all attributes
        attr_list = [
            attributes['speaker'],
            attributes['pitch'],
            attributes['content'],
            attributes['emotion'],
            attributes['energy'],
        ]
        combined = torch.cat(attr_list, dim=-1)

        # Project to prefix tokens
        batch_size = combined.shape[0]
        prefix = self.prefix_projection(combined)
        prefix = prefix.view(batch_size, self.num_prefix_tokens, self.output_dim)

        return {
            'prosody_tokens': prefix,
            'attributes': attributes,
            'original_embeddings': result['original_embeddings'],
            'modified_embeddings': result['modified_embeddings'],
        }

    def transfer_prosody(
        self,
        content_codes: torch.Tensor,
        prosody_codes: torch.Tensor,
        prosody_strength: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Transfer prosody and generate prefix tokens.

        Args:
            content_codes: Source for content/speaker
            prosody_codes: Source for prosody
            prosody_strength: Transfer strength

        Returns:
            Dictionary with transferred prosody tokens
        """
        # Transfer prosody
        modified = self.editor.transfer_prosody(
            content_codes, prosody_codes, prosody_strength
        )

        # Extract attributes from modified embeddings
        # (Need to pass through extractor with modified embeddings)
        attributes = self.editor.extractor(modified)

        # Generate prefix tokens
        attr_list = [
            attributes['speaker'],
            attributes['pitch'],
            attributes['content'],
            attributes['emotion'],
            attributes['energy'],
        ]
        combined = torch.cat(attr_list, dim=-1)

        batch_size = combined.shape[0]
        prefix = self.prefix_projection(combined)
        prefix = prefix.view(batch_size, self.num_prefix_tokens, self.output_dim)

        return {
            'prosody_tokens': prefix,
            'attributes': attributes,
            'modified_embeddings': modified,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def layer_aware_attribute_edit(
    editor: AnCoGenEditor,
    codes: torch.Tensor,
    attribute: str,
    target_value: torch.Tensor,
    layers: Optional[List[int]] = None,
    strength: float = 1.0,
) -> Dict[int, torch.Tensor]:
    """
    Edit attribute at specific layers only.

    Based on AnCoGen finding that different attributes dominate different layers:
    - Content: layers 0-1
    - Speaker: layers 3-7
    - Pitch: layers 2-4

    Args:
        editor: AnCoGenEditor instance
        codes: Input codes
        attribute: Attribute to edit
        target_value: Target attribute value
        layers: Specific layers to edit (None = use config defaults)
        strength: Edit strength

    Returns:
        Modified embeddings
    """
    # Get default layers if not specified
    if layers is None:
        layers = getattr(editor.config, f'{attribute}_layers', list(range(editor.config.num_rvq_layers)))

    # Temporarily override injector's target layers
    injector = editor.injector.injectors[attribute]
    original_layers = injector.target_layers
    injector.target_layers = layers

    try:
        result = editor.edit_attribute(
            codes=codes,
            attribute=attribute,
            target_value=target_value,
            strength=strength,
        )
    finally:
        # Restore original layers
        injector.target_layers = original_layers

    return result


def create_ancogen_editor(
    num_rvq_layers: int = 8,
    embedding_dim: int = 256,
    codebook_size: int = 1024,
    **kwargs,
) -> AnCoGenEditor:
    """
    Factory function to create AnCoGen editor with common settings.

    Args:
        num_rvq_layers: Number of RVQ layers
        embedding_dim: Embedding dimension
        codebook_size: Codebook size
        **kwargs: Additional config parameters

    Returns:
        Configured AnCoGenEditor
    """
    config = AnCoGenConfig(
        num_rvq_layers=num_rvq_layers,
        embedding_dim=embedding_dim,
        codebook_size=codebook_size,
        **kwargs,
    )

    return AnCoGenEditor(config)


def analyze_layer_contributions(
    editor: AnCoGenEditor,
    codes: torch.Tensor,
    attribute: str,
) -> Dict[int, float]:
    """
    Analyze how much each layer contributes to an attribute.

    Uses attention weights from the attribute extractor.

    Args:
        editor: AnCoGenEditor instance
        codes: Input codes
        attribute: Attribute to analyze

    Returns:
        Dictionary mapping layer index to contribution weight
    """
    layer_embeddings = editor.codes_to_embeddings(codes)
    extractor = editor.extractor.extractors[attribute]

    # Get source layer embeddings
    source_embs = []
    for layer_idx in extractor.source_layers:
        if layer_idx in layer_embeddings:
            source_embs.append(layer_embeddings[layer_idx])

    # Stack and get attention weights
    stacked = torch.stack(source_embs, dim=-2)  # [batch, seq, num_layers, dim]
    attn_weights = extractor.layer_attention(stacked)  # [batch, seq, num_layers, 1]

    # Average over batch and sequence
    avg_weights = attn_weights.mean(dim=(0, 1)).squeeze(-1)  # [num_layers]

    return {
        layer_idx: avg_weights[i].item()
        for i, layer_idx in enumerate(extractor.source_layers)
    }


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'AnCoGenConfig',
    'AnCoGenEditor',
    'AnCoGenLoss',
    'AnCoGenAdapter',
    'AttributeExtractor',
    'AttributeInjector',
    'MultiAttributeExtractor',
    'MultiAttributeInjector',
    'layer_aware_attribute_edit',
    'create_ancogen_editor',
    'analyze_layer_contributions',
]
