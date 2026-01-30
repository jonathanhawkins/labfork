#!/usr/bin/env python3
"""
VoxCPM Inference Script

Generate speech using VoxCPM with zero-shot voice cloning.

Usage:
    # Basic generation
    python generate_with_voxcpm.py \
        --text "Hello, how are you?" \
        --checkpoint ../checkpoints/voxcpm/best.pt \
        --output hello.wav

    # Voice cloning from reference
    python generate_with_voxcpm.py \
        --text "Hello, how are you?" \
        --reference reference.wav \
        --checkpoint ../checkpoints/voxcpm/best.pt \
        --output cloned_hello.wav

    # Adjust generation parameters
    python generate_with_voxcpm.py \
        --text "Hello, how are you?" \
        --reference reference.wav \
        --checkpoint ../checkpoints/voxcpm/best.pt \
        --num-steps 25 \
        --cfg-scale 3.0 \
        --output fast_hello.wav

    # Streaming mode
    python generate_with_voxcpm.py \
        --text "Hello, how are you?" \
        --reference reference.wav \
        --checkpoint ../checkpoints/voxcpm/best.pt \
        --streaming \
        --output stream_hello.wav

Based on arXiv:2509.24650:
"VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation
and True-to-Life Voice Cloning"
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

from voxcpm import (
    VoxCPMConfig,
    VoxCPM,
    VoxCPMAdapter,
    estimate_rtf,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoxCPMGenerator:
    """VoxCPM speech generator with voice cloning support."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        use_torch_compile: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Create config from checkpoint
        config_dict = checkpoint.get('config', {})
        self.config = VoxCPMConfig(**config_dict) if config_dict else VoxCPMConfig()

        # Create model
        self.model = VoxCPM(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Optionally compile model
        if use_torch_compile and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode='reduce-overhead')

        # Build simple vocabulary
        self.build_vocab()

        logger.info(f"Model loaded on {self.device}")

    def build_vocab(self):
        """Build character vocabulary for text tokenization."""
        # Basic ASCII + common Unicode
        chars = list(' !"#$%&\'()*+,-./0123456789:;<=>?@')
        chars += list('ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`')
        chars += list('abcdefghijklmnopqrstuvwxyz{|}~')
        chars += ['<pad>', '<unk>', '<bos>', '<eos>']

        self.char_to_idx = {c: i for i, c in enumerate(chars)}

    def text_to_tokens(self, text: str) -> torch.Tensor:
        """Convert text to token indices."""
        tokens = []
        for c in text:
            if c in self.char_to_idx:
                tokens.append(self.char_to_idx[c])
            else:
                tokens.append(self.char_to_idx.get('<unk>', 0))

        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

    def load_reference(self, path: str) -> torch.Tensor:
        """Load and preprocess reference audio."""
        audio, sr = torchaudio.load(path)

        # Resample if needed
        if sr != self.config.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.config.sample_rate)

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Add batch dimension
        audio = audio.unsqueeze(0)  # [1, 1, samples]

        return audio.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        cfg_scale: float = 2.0,
        max_len: int = 500,
    ) -> torch.Tensor:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize
            reference_audio: Optional reference for voice cloning [1, 1, samples]
            num_steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            max_len: Maximum generation length (frames)

        Returns:
            Generated audio [1, 1, samples]
        """
        # Tokenize text
        text_tokens = self.text_to_tokens(text)
        text_mask = torch.ones(1, text_tokens.shape[1], dtype=torch.bool, device=self.device)

        # Generate
        audio = self.model.generate(
            text_tokens=text_tokens,
            reference_audio=reference_audio,
            text_mask=text_mask,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            max_len=max_len,
        )

        return audio

    @torch.no_grad()
    def generate_streaming(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        cfg_scale: float = 2.0,
        chunk_frames: int = 50,
    ):
        """
        Generate speech in streaming mode (chunk by chunk).

        Yields audio chunks as they are generated.

        Args:
            text: Input text to synthesize
            reference_audio: Optional reference for voice cloning
            num_steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            chunk_frames: Frames per chunk

        Yields:
            Audio chunks [1, 1, samples_per_chunk]
        """
        # Tokenize text
        text_tokens = self.text_to_tokens(text)
        text_mask = torch.ones(1, text_tokens.shape[1], dtype=torch.bool, device=self.device)

        # Encode reference if provided
        if reference_audio is not None:
            ref_latent = self.model.vae.encode(reference_audio)['latent']
        else:
            ref_latent = None

        # Get TSLM output
        tslm_out = self.model.tslm(text_tokens, text_mask, ref_latent)

        # Estimate total frames
        total_frames = text_tokens.shape[1] * 4
        num_chunks = (total_frames + chunk_frames - 1) // chunk_frames

        # Generate chunk by chunk
        previous_latent = ref_latent[:, :10] if ref_latent is not None else None

        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * chunk_frames
            end_frame = min(start_frame + chunk_frames, total_frames)
            chunk_len = end_frame - start_frame

            # Get TSLM chunk
            tslm_chunk = tslm_out['hidden'][:, start_frame:end_frame]

            # FSQ quantize
            fsq_out = self.model.fsq(tslm_chunk)

            # RALM with previous context
            ralm_out = self.model.ralm(
                tslm_out['text_hidden'],
                fsq_out['quantized'],
                previous_latent,
            )

            # Combine conditioning
            combined = torch.cat([
                fsq_out['quantized'],
                ralm_out['residual'],
            ], dim=-1)
            conditioning = self.model.condition_proj(combined)

            # Sample latent
            latent = self.model.sample(
                conditioning,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
            )

            # Decode to audio
            chunk_audio = self.model.vae.decode(latent)

            # Update context for next chunk
            previous_latent = latent

            yield chunk_audio

    def measure_rtf(
        self,
        text_length: int = 50,
        audio_duration: float = 5.0,
        num_runs: int = 5,
    ) -> float:
        """Measure Real-Time Factor."""
        return estimate_rtf(
            self.model,
            text_length=text_length,
            audio_duration=audio_duration,
            device=str(self.device),
            num_runs=num_runs,
        )


def main():
    parser = argparse.ArgumentParser(description="Generate speech with VoxCPM")
    parser.add_argument('--text', type=str, required=True,
                        help='Text to synthesize')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to reference audio for voice cloning')
    parser.add_argument('--output', type=str, default='output.wav',
                        help='Output audio file path')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='Number of diffusion steps')
    parser.add_argument('--cfg-scale', type=float, default=2.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--max-len', type=int, default=500,
                        help='Maximum generation length (frames)')
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming generation')
    parser.add_argument('--measure-rtf', action='store_true',
                        help='Measure and report RTF')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for faster inference')
    args = parser.parse_args()

    # Create generator
    generator = VoxCPMGenerator(
        args.checkpoint,
        device=args.device,
        use_torch_compile=args.compile,
    )

    # Load reference if provided
    reference_audio = None
    if args.reference:
        logger.info(f"Loading reference audio from {args.reference}")
        reference_audio = generator.load_reference(args.reference)
        logger.info(f"Reference audio shape: {reference_audio.shape}")

    # Measure RTF if requested
    if args.measure_rtf:
        logger.info("Measuring Real-Time Factor...")
        rtf = generator.measure_rtf()
        logger.info(f"RTF: {rtf:.3f} ({'faster' if rtf < 1 else 'slower'} than realtime)")
        logger.info(f"Speed: {1/rtf:.1f}x realtime")

    # Generate
    logger.info(f"Generating speech for: '{args.text}'")
    start_time = time.time()

    if args.streaming:
        # Streaming generation
        audio_chunks = []
        for chunk in generator.generate_streaming(
            args.text,
            reference_audio=reference_audio,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
        ):
            audio_chunks.append(chunk)

        audio = torch.cat(audio_chunks, dim=-1)
    else:
        # Single-shot generation
        audio = generator.generate(
            args.text,
            reference_audio=reference_audio,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            max_len=args.max_len,
        )

    generation_time = time.time() - start_time
    audio_duration = audio.shape[-1] / generator.config.sample_rate

    logger.info(f"Generated {audio_duration:.2f}s of audio in {generation_time:.2f}s")
    logger.info(f"RTF: {generation_time / audio_duration:.3f}")

    # Save output
    audio = audio.squeeze(0).cpu()  # Remove batch dim
    torchaudio.save(args.output, audio, generator.config.sample_rate)
    logger.info(f"Saved audio to {args.output}")


if __name__ == '__main__':
    main()
