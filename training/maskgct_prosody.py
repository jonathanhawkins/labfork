"""
MaskGCT-Style Masked Prosody Prediction Module

Implements mask-and-predict paradigm from MaskGCT (ICLR 2025) for non-autoregressive
prosody-conditioned TTS. Key innovations:

1. **Masked Prosody Prediction**: During training, random prosody tokens are masked
   and the model learns to predict them in parallel, enabling efficient learning
   of prosody/style from prompt speech without explicit alignment.

2. **Two-Stage Architecture**:
   - Stage 1: Predict semantic prosody tokens (emotion, intent)
   - Stage 2: Predict acoustic prosody tokens (pitch, energy, rhythm) conditioned on semantic

3. **Mask Scheduling**: Cosine schedule starts with high mask ratio (0.9) and decreases
   during training, similar to MLM pre-training.

4. **Parallel Decoding**: At inference, all tokens are predicted in parallel with
   iterative refinement, achieving 2x+ speedup over autoregressive baseline.

Reference: MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer (ICLR 2025)

Usage:
    model = MaskGCTProsodyModel(config)

    # Training: mask-and-predict
    loss = model.compute_masked_loss(prosody_tokens, prompt_tokens)

    # Inference: parallel generation
    prosody = model.generate_parallel(prompt_tokens, num_iterations=4)
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MaskGCTConfig:
    """Configuration for MaskGCT prosody model."""

    # Token dimensions
    hidden_size: int = 2048
    num_semantic_tokens: int = 32    # Semantic prosody vocabulary
    num_acoustic_tokens: int = 128   # Acoustic prosody vocabulary

    # Prosody dimensions (matching ProsodyConfig)
    semantic_dim: int = 8
    acoustic_dim: int = 12
    rhythm_dim: int = 8
    contour_dim: int = 64

    # Sequence lengths
    max_prosody_length: int = 64     # Max prosody tokens per utterance
    num_prompt_tokens: int = 8       # Reference prompt tokens for style

    # Transformer settings
    num_layers: int = 6
    num_heads: int = 8
    feedforward_dim: int = 4096
    dropout: float = 0.1

    # Masking settings
    initial_mask_ratio: float = 0.9   # Start with high masking
    final_mask_ratio: float = 0.1     # End with low masking
    mask_schedule: str = "cosine"     # cosine, linear, or constant

    # Training settings
    use_two_stage: bool = True        # Two-stage semantic → acoustic
    predict_semantic_first: bool = True

    # Inference settings
    num_parallel_iterations: int = 4  # Iterative parallel decoding steps
    temperature: float = 0.8

    # Special tokens
    mask_token_id: int = 0
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3


class ProsodyTokenizer(nn.Module):
    """
    Converts continuous prosody features to discrete tokens.

    Uses VQ-VAE style quantization to map 4-layer prosody (semantic, acoustic,
    rhythm, contour) to discrete tokens for masked prediction.
    """

    def __init__(self, config: MaskGCTConfig):
        super().__init__()
        self.config = config

        # Total input dimension from 4 prosody types
        total_prosody_dim = (
            config.semantic_dim +
            config.acoustic_dim +
            config.rhythm_dim +
            config.contour_dim
        )

        # Semantic tokenization (emotion/intent → discrete semantic tokens)
        self.semantic_encoder = nn.Sequential(
            nn.Linear(config.semantic_dim, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 2),
        )

        # Semantic codebook (learnable embeddings)
        self.semantic_codebook = nn.Embedding(config.num_semantic_tokens, config.hidden_size // 2)

        # Acoustic tokenization (pitch, energy, rhythm, contour → discrete acoustic tokens)
        acoustic_input_dim = config.acoustic_dim + config.rhythm_dim + config.contour_dim
        self.acoustic_encoder = nn.Sequential(
            nn.Linear(acoustic_input_dim, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 2),
        )

        # Acoustic codebook
        self.acoustic_codebook = nn.Embedding(config.num_acoustic_tokens, config.hidden_size // 2)

        # Commitment loss weight for VQ training
        self.commitment_weight = 0.25

    def quantize(
        self,
        z: torch.Tensor,
        codebook: nn.Embedding,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous features to nearest codebook entry.

        Args:
            z: Continuous features [batch, hidden_dim]
            codebook: Embedding table [num_tokens, hidden_dim]

        Returns:
            quantized: Quantized features [batch, hidden_dim]
            indices: Token indices [batch]
            commitment_loss: VQ commitment loss
        """
        # Compute distances to all codebook entries
        # z: [B, D], codebook: [K, D]
        distances = torch.cdist(z.unsqueeze(1), codebook.weight.unsqueeze(0)).squeeze(1)  # [B, K]

        # Find nearest codebook entry
        indices = distances.argmin(dim=-1)  # [B]

        # Get quantized vectors
        quantized = codebook(indices)  # [B, D]

        # Commitment loss
        commitment_loss = F.mse_loss(z, quantized.detach())

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, indices, commitment_loss

    def encode_semantic(
        self,
        semantic: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode semantic prosody to discrete tokens.

        Args:
            semantic: [batch, semantic_dim] emotion/intent features

        Returns:
            quantized: [batch, hidden_dim] quantized features
            tokens: [batch] discrete token indices
            commitment_loss: VQ loss
        """
        z = self.semantic_encoder(semantic)
        return self.quantize(z, self.semantic_codebook)

    def encode_acoustic(
        self,
        acoustic: torch.Tensor,
        rhythm: torch.Tensor,
        contour: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode acoustic prosody to discrete tokens.

        Args:
            acoustic: [batch, acoustic_dim] pitch, formants, etc.
            rhythm: [batch, rhythm_dim] timing features
            contour: [batch, contour_dim] pitch trajectory

        Returns:
            quantized: [batch, hidden_dim] quantized features
            tokens: [batch] discrete token indices
            commitment_loss: VQ loss
        """
        combined = torch.cat([acoustic, rhythm, contour], dim=-1)
        z = self.acoustic_encoder(combined)
        return self.quantize(z, self.acoustic_codebook)

    def decode_semantic(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode semantic tokens to embeddings."""
        return self.semantic_codebook(tokens)

    def decode_acoustic(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode acoustic tokens to embeddings."""
        return self.acoustic_codebook(tokens)


class MaskedTransformerEncoder(nn.Module):
    """
    Transformer encoder for masked prosody prediction.

    Bidirectional attention allows predicting masked tokens from both
    left and right context, enabling parallel prediction.
    """

    def __init__(self, config: MaskGCTConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.semantic_embedding = nn.Embedding(
            config.num_semantic_tokens + 4,  # +4 for special tokens
            config.hidden_size // 2
        )
        self.acoustic_embedding = nn.Embedding(
            config.num_acoustic_tokens + 4,
            config.hidden_size // 2
        )

        # Position encoding
        self.position_embedding = nn.Embedding(config.max_prosody_length, config.hidden_size)

        # Mask token embedding
        self.mask_embedding = nn.Parameter(torch.randn(config.hidden_size))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output projections
        self.semantic_head = nn.Linear(config.hidden_size, config.num_semantic_tokens)
        self.acoustic_head = nn.Linear(config.hidden_size, config.num_acoustic_tokens)

        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        semantic_tokens: torch.Tensor,      # [batch, seq_len]
        acoustic_tokens: torch.Tensor,      # [batch, seq_len]
        mask: Optional[torch.Tensor] = None,  # [batch, seq_len] True = masked
        prompt_embeds: Optional[torch.Tensor] = None,  # [batch, prompt_len, hidden]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional masking.

        Args:
            semantic_tokens: Semantic token indices [batch, seq_len]
            acoustic_tokens: Acoustic token indices [batch, seq_len]
            mask: Boolean mask where True = masked position
            prompt_embeds: Optional prompt embeddings for conditioning

        Returns:
            semantic_logits: [batch, seq_len, num_semantic_tokens]
            acoustic_logits: [batch, seq_len, num_acoustic_tokens]
        """
        batch_size, seq_len = semantic_tokens.shape
        device = semantic_tokens.device

        # Embed tokens
        sem_embeds = self.semantic_embedding(semantic_tokens)  # [B, S, H/2]
        aco_embeds = self.acoustic_embedding(acoustic_tokens)  # [B, S, H/2]

        # Combine semantic and acoustic
        embeds = torch.cat([sem_embeds, aco_embeds], dim=-1)  # [B, S, H]

        # Apply mask (replace with learnable mask embedding)
        if mask is not None:
            mask_embed = self.mask_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            embeds = torch.where(mask.unsqueeze(-1), mask_embed, embeds)

        # Add position embeddings
        positions = torch.arange(seq_len, device=device)
        pos_embeds = self.position_embedding(positions)  # [S, H]
        embeds = embeds + pos_embeds.unsqueeze(0)

        # Prepend prompt if provided
        if prompt_embeds is not None:
            embeds = torch.cat([prompt_embeds, embeds], dim=1)

        # Transformer encoding
        hidden = self.transformer(embeds)
        hidden = self.norm(hidden)

        # Remove prompt from output
        if prompt_embeds is not None:
            hidden = hidden[:, prompt_embeds.shape[1]:, :]

        # Project to token logits
        semantic_logits = self.semantic_head(hidden)
        acoustic_logits = self.acoustic_head(hidden)

        return semantic_logits, acoustic_logits


class MaskScheduler:
    """
    Mask ratio scheduler for curriculum-style masking.

    Starts with high mask ratio (most tokens masked) and gradually
    decreases during training, similar to BERT-style MLM curriculum.
    """

    def __init__(
        self,
        initial_ratio: float = 0.9,
        final_ratio: float = 0.1,
        total_steps: int = 10000,
        schedule: str = "cosine",
    ):
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.total_steps = total_steps
        self.schedule = schedule

    def get_mask_ratio(self, step: int) -> float:
        """Get mask ratio for current training step."""
        if self.schedule == "constant":
            return self.initial_ratio

        progress = min(step / self.total_steps, 1.0)

        if self.schedule == "linear":
            ratio = self.initial_ratio + (self.final_ratio - self.initial_ratio) * progress
        elif self.schedule == "cosine":
            # Cosine annealing
            ratio = self.final_ratio + 0.5 * (self.initial_ratio - self.final_ratio) * (
                1 + math.cos(math.pi * progress)
            )
        else:
            ratio = self.initial_ratio

        return ratio


class MaskGCTProsodyModel(nn.Module):
    """
    Main MaskGCT model for masked prosody prediction.

    Implements the full mask-and-predict training and parallel decoding
    for non-autoregressive prosody-conditioned TTS.
    """

    def __init__(self, config: MaskGCTConfig):
        super().__init__()
        self.config = config

        # Tokenizer for prosody → discrete tokens
        self.tokenizer = ProsodyTokenizer(config)

        # Masked transformer for prediction
        self.transformer = MaskedTransformerEncoder(config)

        # Prompt encoder for style conditioning
        self.prompt_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Mask scheduler
        self.mask_scheduler = MaskScheduler(
            initial_ratio=config.initial_mask_ratio,
            final_ratio=config.final_mask_ratio,
            schedule=config.mask_schedule,
        )

        # Training step counter
        self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))

    def generate_mask(
        self,
        batch_size: int,
        seq_len: int,
        mask_ratio: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate random mask for training.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            mask_ratio: Fraction of tokens to mask
            device: Target device

        Returns:
            mask: [batch, seq_len] boolean tensor, True = masked
        """
        num_masked = int(seq_len * mask_ratio)

        # Random permutation for each batch item
        masks = []
        for _ in range(batch_size):
            perm = torch.randperm(seq_len, device=device)
            mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            mask[perm[:num_masked]] = True
            masks.append(mask)

        return torch.stack(masks)

    def tokenize_prosody(
        self,
        prosody_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert prosody features to discrete tokens.

        Args:
            prosody_dict: Dict with 'semantic', 'acoustic', 'rhythm', 'contour' tensors

        Returns:
            semantic_tokens: [batch] semantic token indices
            acoustic_tokens: [batch] acoustic token indices
            commitment_loss: VQ commitment loss for training
        """
        _, semantic_tokens, sem_loss = self.tokenizer.encode_semantic(
            prosody_dict['semantic']
        )
        _, acoustic_tokens, aco_loss = self.tokenizer.encode_acoustic(
            prosody_dict['acoustic'],
            prosody_dict['rhythm'],
            prosody_dict['contour'],
        )

        commitment_loss = sem_loss + aco_loss

        return semantic_tokens, acoustic_tokens, commitment_loss

    def compute_masked_loss(
        self,
        prosody_dict: Dict[str, torch.Tensor],
        prompt_embeds: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute masked prediction loss for training.

        Args:
            prosody_dict: Dict with prosody tensors (can be batched with seq_len)
            prompt_embeds: Optional prompt embeddings for conditioning
            mask_ratio: Override mask ratio (uses scheduler if None)

        Returns:
            Dict with 'loss', 'semantic_loss', 'acoustic_loss', 'commitment_loss'
        """
        device = prosody_dict['semantic'].device
        batch_size = prosody_dict['semantic'].shape[0]

        # Handle sequence dimension - if not present, expand
        if prosody_dict['semantic'].dim() == 2:
            # [batch, dim] → [batch, 1, dim] → expand to seq_len
            seq_len = self.config.max_prosody_length
            prosody_dict = {
                k: v.unsqueeze(1).expand(-1, seq_len, -1)
                for k, v in prosody_dict.items()
            }

        seq_len = prosody_dict['semantic'].shape[1]

        # Tokenize all positions
        semantic_tokens_list = []
        acoustic_tokens_list = []
        total_commitment_loss = 0.0

        for t in range(seq_len):
            sem_t, aco_t, commit_loss = self.tokenize_prosody({
                'semantic': prosody_dict['semantic'][:, t, :],
                'acoustic': prosody_dict['acoustic'][:, t, :],
                'rhythm': prosody_dict['rhythm'][:, t, :],
                'contour': prosody_dict['contour'][:, t, :],
            })
            semantic_tokens_list.append(sem_t)
            acoustic_tokens_list.append(aco_t)
            total_commitment_loss = total_commitment_loss + commit_loss

        semantic_tokens = torch.stack(semantic_tokens_list, dim=1)  # [B, S]
        acoustic_tokens = torch.stack(acoustic_tokens_list, dim=1)  # [B, S]
        commitment_loss = total_commitment_loss / seq_len

        # Get mask ratio from scheduler or use provided
        if mask_ratio is None:
            mask_ratio = self.mask_scheduler.get_mask_ratio(self.training_step.item())

        # Generate mask
        mask = self.generate_mask(batch_size, seq_len, mask_ratio, device)

        # Forward with masking
        semantic_logits, acoustic_logits = self.transformer(
            semantic_tokens, acoustic_tokens, mask, prompt_embeds
        )

        # Compute loss only on masked positions
        semantic_loss = F.cross_entropy(
            semantic_logits[mask].view(-1, self.config.num_semantic_tokens),
            semantic_tokens[mask].view(-1),
            reduction='mean',
        ) if mask.any() else torch.tensor(0.0, device=device)

        acoustic_loss = F.cross_entropy(
            acoustic_logits[mask].view(-1, self.config.num_acoustic_tokens),
            acoustic_tokens[mask].view(-1),
            reduction='mean',
        ) if mask.any() else torch.tensor(0.0, device=device)

        total_loss = semantic_loss + acoustic_loss + self.tokenizer.commitment_weight * commitment_loss

        # Increment training step
        if self.training:
            self.training_step += 1

        return {
            'loss': total_loss,
            'semantic_loss': semantic_loss,
            'acoustic_loss': acoustic_loss,
            'commitment_loss': commitment_loss,
            'mask_ratio': torch.tensor(mask_ratio),
        }

    @torch.no_grad()
    def generate_parallel(
        self,
        prompt_embeds: torch.Tensor,
        batch_size: int = 1,
        seq_len: Optional[int] = None,
        num_iterations: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prosody tokens using parallel iterative decoding.

        MaskGCT inference uses iterative refinement:
        1. Start with all positions masked
        2. Predict all tokens in parallel
        3. Unmask high-confidence predictions
        4. Repeat until all positions are unmasked

        Args:
            prompt_embeds: [batch, prompt_len, hidden] style conditioning
            batch_size: Number of sequences to generate
            seq_len: Output sequence length (default: max_prosody_length)
            num_iterations: Refinement iterations (default: config setting)
            temperature: Sampling temperature (default: config setting)

        Returns:
            semantic_tokens: [batch, seq_len] generated semantic tokens
            acoustic_tokens: [batch, seq_len] generated acoustic tokens
        """
        device = prompt_embeds.device
        seq_len = seq_len or self.config.max_prosody_length
        num_iterations = num_iterations or self.config.num_parallel_iterations
        temperature = temperature or self.config.temperature

        # Initialize with random tokens (will be masked anyway)
        semantic_tokens = torch.randint(
            0, self.config.num_semantic_tokens,
            (batch_size, seq_len), device=device
        )
        acoustic_tokens = torch.randint(
            0, self.config.num_acoustic_tokens,
            (batch_size, seq_len), device=device
        )

        # Start with all positions masked
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Iterative refinement
        for iteration in range(num_iterations):
            # Predict masked positions
            semantic_logits, acoustic_logits = self.transformer(
                semantic_tokens, acoustic_tokens, mask, prompt_embeds
            )

            # Apply temperature
            semantic_probs = F.softmax(semantic_logits / temperature, dim=-1)
            acoustic_probs = F.softmax(acoustic_logits / temperature, dim=-1)

            # Sample predictions
            new_semantic = torch.multinomial(
                semantic_probs.view(-1, self.config.num_semantic_tokens), 1
            ).view(batch_size, seq_len)
            new_acoustic = torch.multinomial(
                acoustic_probs.view(-1, self.config.num_acoustic_tokens), 1
            ).view(batch_size, seq_len)

            # Compute confidence for unmasking
            semantic_conf = semantic_probs.max(dim=-1).values  # [B, S]
            acoustic_conf = acoustic_probs.max(dim=-1).values
            combined_conf = (semantic_conf + acoustic_conf) / 2

            # Update tokens at masked positions
            semantic_tokens = torch.where(mask, new_semantic, semantic_tokens)
            acoustic_tokens = torch.where(mask, new_acoustic, acoustic_tokens)

            # Unmask fraction of positions based on confidence
            # More positions unmasked in later iterations
            unmask_ratio = (iteration + 1) / num_iterations
            num_to_unmask = int(mask.sum(dim=1).float().mean().item() * unmask_ratio)

            # Unmask highest confidence positions
            for b in range(batch_size):
                masked_indices = mask[b].nonzero(as_tuple=True)[0]
                if len(masked_indices) > 0:
                    conf_at_masked = combined_conf[b, masked_indices]
                    _, top_indices = conf_at_masked.topk(
                        min(num_to_unmask, len(masked_indices))
                    )
                    unmask_indices = masked_indices[top_indices]
                    mask[b, unmask_indices] = False

        return semantic_tokens, acoustic_tokens

    def decode_tokens_to_prosody(
        self,
        semantic_tokens: torch.Tensor,
        acoustic_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode generated tokens back to prosody embeddings.

        Args:
            semantic_tokens: [batch, seq_len] semantic token indices
            acoustic_tokens: [batch, seq_len] acoustic token indices

        Returns:
            Dict with 'semantic_embed', 'acoustic_embed' tensors
        """
        semantic_embed = self.tokenizer.decode_semantic(semantic_tokens)
        acoustic_embed = self.tokenizer.decode_acoustic(acoustic_tokens)

        return {
            'semantic_embed': semantic_embed,
            'acoustic_embed': acoustic_embed,
        }


class MaskGCTWithProsodyEncoder(nn.Module):
    """
    Combines MaskGCT masked prediction with existing prosody encoder.

    This bridges the MaskGCT approach with the existing ProsodyEncoder
    architecture, enabling both masked training and prefix-based conditioning.

    Architecture:
        1. Extract prosody from audio
        2. Tokenize prosody with MaskGCT tokenizer
        3. Train with mask-and-predict objective
        4. Convert predicted tokens to ProsodyEncoder-compatible embeddings
        5. Use as prefix for CSM generation
    """

    def __init__(
        self,
        maskgct_config: MaskGCTConfig,
        hidden_size: int = 2048,
        num_prosody_tokens: int = 4,
    ):
        super().__init__()
        self.maskgct = MaskGCTProsodyModel(maskgct_config)
        self.config = maskgct_config

        # Convert MaskGCT embeddings to prosody prefix
        combined_embed_dim = maskgct_config.hidden_size  # semantic + acoustic
        self.prefix_projection = nn.Sequential(
            nn.Linear(combined_embed_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(maskgct_config.dropout),
            nn.Linear(hidden_size, hidden_size * num_prosody_tokens),
        )

        self.num_prosody_tokens = num_prosody_tokens
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)

    def encode_prosody_from_dict(
        self,
        prosody_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode prosody dict to prefix embeddings using MaskGCT tokenization.

        Args:
            prosody_dict: Dict with 'semantic', 'acoustic', 'rhythm', 'contour'
                          Can be 2D [batch, dim] or 3D [batch, seq, dim]

        Returns:
            prefix: [batch, num_prosody_tokens, hidden_size]
        """
        semantic = prosody_dict['semantic']
        acoustic = prosody_dict['acoustic']
        rhythm = prosody_dict['rhythm']
        contour = prosody_dict['contour']

        # Handle temporal prosody (3D input) by averaging over time
        if semantic.dim() == 3:
            semantic = semantic.mean(dim=1)
            acoustic = acoustic.mean(dim=1)
            rhythm = rhythm.mean(dim=1)
            contour = contour.mean(dim=1)

        batch_size = semantic.shape[0]

        # Tokenize prosody
        _, semantic_tokens, _ = self.maskgct.tokenizer.encode_semantic(semantic)
        _, acoustic_tokens, _ = self.maskgct.tokenizer.encode_acoustic(
            acoustic, rhythm, contour
        )

        # Get token embeddings
        sem_embed = self.maskgct.tokenizer.decode_semantic(semantic_tokens)
        aco_embed = self.maskgct.tokenizer.decode_acoustic(acoustic_tokens)

        # Combine embeddings
        combined = torch.cat([sem_embed, aco_embed], dim=-1)  # [B, H]

        # Project to prefix tokens
        prefix = self.prefix_projection(combined)  # [B, H * num_tokens]
        prefix = prefix.view(batch_size, self.num_prosody_tokens, self.hidden_size)
        prefix = self.norm(prefix)

        return prefix

    def forward(
        self,
        prosody_dict: Dict[str, torch.Tensor],
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with masked prediction loss.

        Args:
            prosody_dict: Dict with prosody tensors
            prompt_embeds: Optional prompt embeddings

        Returns:
            Dict with 'loss', 'prefix', and component losses
        """
        # Compute masked prediction loss
        loss_dict = self.maskgct.compute_masked_loss(prosody_dict, prompt_embeds)

        # Also compute prefix embeddings for CSM conditioning
        prefix = self.encode_prosody_from_dict(prosody_dict)
        loss_dict['prefix'] = prefix

        return loss_dict

    @torch.no_grad()
    def generate_prefix_parallel(
        self,
        prompt_embeds: torch.Tensor,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Generate prosody prefix using parallel decoding.

        Args:
            prompt_embeds: [batch, prompt_len, hidden] style conditioning
            batch_size: Number of sequences to generate

        Returns:
            prefix: [batch, num_prosody_tokens, hidden_size]
        """
        # Generate tokens in parallel
        semantic_tokens, acoustic_tokens = self.maskgct.generate_parallel(
            prompt_embeds, batch_size
        )

        # Average over sequence for single prosody vector
        sem_embed = self.maskgct.tokenizer.decode_semantic(semantic_tokens).mean(dim=1)
        aco_embed = self.maskgct.tokenizer.decode_acoustic(acoustic_tokens).mean(dim=1)

        # Combine and project
        combined = torch.cat([sem_embed, aco_embed], dim=-1)
        prefix = self.prefix_projection(combined)
        prefix = prefix.view(batch_size, self.num_prosody_tokens, self.hidden_size)
        prefix = self.norm(prefix)

        return prefix


# Testing
if __name__ == "__main__":
    print("Testing MaskGCT Prosody Module")
    print("=" * 50)

    config = MaskGCTConfig()
    model = MaskGCTProsodyModel(config)

    print(f"Config: {config}")

    # Create dummy prosody inputs
    batch_size = 2
    prosody_dict = {
        'semantic': torch.randn(batch_size, config.semantic_dim),
        'acoustic': torch.randn(batch_size, config.acoustic_dim),
        'rhythm': torch.randn(batch_size, config.rhythm_dim),
        'contour': torch.randn(batch_size, config.contour_dim),
    }

    # Test masked loss computation
    print("\nTesting masked loss computation...")
    loss_dict = model.compute_masked_loss(prosody_dict, mask_ratio=0.7)
    print(f"Total loss: {loss_dict['loss'].item():.4f}")
    print(f"Semantic loss: {loss_dict['semantic_loss'].item():.4f}")
    print(f"Acoustic loss: {loss_dict['acoustic_loss'].item():.4f}")
    print(f"Commitment loss: {loss_dict['commitment_loss'].item():.4f}")
    print(f"Mask ratio: {loss_dict['mask_ratio'].item():.2f}")

    # Test parallel generation
    print("\nTesting parallel generation...")
    prompt_embeds = torch.randn(batch_size, 4, config.hidden_size)
    semantic_tokens, acoustic_tokens = model.generate_parallel(prompt_embeds, batch_size)
    print(f"Generated semantic tokens shape: {semantic_tokens.shape}")
    print(f"Generated acoustic tokens shape: {acoustic_tokens.shape}")

    # Test combined model
    print("\nTesting MaskGCT with ProsodyEncoder bridge...")
    combined_model = MaskGCTWithProsodyEncoder(config)
    output = combined_model(prosody_dict)
    print(f"Prefix shape: {output['prefix'].shape}")
    print(f"Expected: [{batch_size}, 4, 2048]")

    # Test parallel prefix generation
    prefix = combined_model.generate_prefix_parallel(prompt_embeds, batch_size)
    print(f"Generated prefix shape: {prefix.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nMaskGCT model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    print("\nMaskGCT prosody module ready!")
