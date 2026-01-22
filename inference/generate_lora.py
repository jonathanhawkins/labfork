"""
Voice Clone Pipeline - LoRA Speech Generation
Generate speech using a LoRA fine-tuned CSM model.

Usage:
    python generate_lora.py --adapter ../models/checkpoints/csm_lora_450/best --text "Hello world!"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torchaudio

# Add parent to path for generator imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class LoRAVoiceGenerator:
    """
    Generate speech using a LoRA fine-tuned CSM model.
    """

    def __init__(
        self,
        adapter_path: str,
        base_model_path: str = "../models/csm-1b",
        device: str = "auto"
    ):
        self.adapter_path = Path(adapter_path)
        self.base_model_path = Path(base_model_path)

        # Setup device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model
        self.model, self.processor = self._load_model()

    def _load_model(self):
        """Load the base model with LoRA adapter."""
        from peft import PeftModel
        from transformers import CsmForConditionalGeneration, AutoProcessor

        # Convert to absolute path
        model_path = self.base_model_path.resolve()
        print(f"Loading base model from: {model_path}")

        # Load CSM using CsmForConditionalGeneration (same as training)
        base_model = CsmForConditionalGeneration.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Match training precision
            local_files_only=True,
        )
        base_model = base_model.to(self.device)
        print(f"Loaded CsmForConditionalGeneration")

        # Load processor
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"Loaded processor")

        print(f"Loading LoRA adapter from: {self.adapter_path}")

        # Load LoRA adapter
        model = PeftModel.from_pretrained(
            base_model,
            str(self.adapter_path),
            is_trainable=False,
        )
        model.eval()

        # Merge LoRA weights for faster inference (optional but recommended)
        # model = model.merge_and_unload()

        print(f"LoRA adapter loaded!")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"LoRA parameters: {trainable_params:,}")

        return model, processor

    def generate(
        self,
        text: str,
        speaker: int = 0,
        temperature: float = 0.8,
        top_k: int = 50,
        max_audio_length_ms: int = 10000,
    ) -> torch.Tensor:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            speaker: Speaker ID
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            max_audio_length_ms: Maximum audio length in milliseconds

        Returns:
            Audio waveform tensor at 24kHz
        """
        print(f"Generating: '{text}'")

        with torch.no_grad():
            try:
                # Build conversation for CSM
                conversation = [
                    {"role": str(speaker), "content": [{"type": "text", "text": text}]}
                ]

                # Use processor to prepare inputs
                inputs = self.processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_dict=True,
                )
                inputs = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                print(f"Input keys: {inputs.keys()}")

                # Generate with CSM
                output = self.model.generate(
                    **inputs,
                    output_audio=True,
                    max_new_tokens=max_audio_length_ms // 80,  # ~80ms per token
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                )

                # Extract audio from output
                # CSM returns a list with one tensor per batch item
                if isinstance(output, list) and len(output) > 0:
                    audio = output[0]  # First batch item
                    if isinstance(audio, torch.Tensor):
                        print(f"Got audio tensor from list, shape: {audio.shape}")
                    else:
                        print(f"List element is {type(audio)}, not tensor")
                        audio = torch.zeros(24000 * 2)
                elif hasattr(output, 'audio') and output.audio is not None:
                    audio = output.audio[0]
                    print(f"Got audio from output.audio")
                elif hasattr(output, 'sequences'):
                    print(f"Output has sequences, shape: {output.sequences.shape}")
                    audio = self._decode_audio(output.sequences)
                elif isinstance(output, torch.Tensor):
                    audio = output
                    print(f"Got audio tensor directly, shape: {audio.shape}")
                else:
                    print(f"Unexpected output type: {type(output)}")
                    audio = torch.zeros(24000 * 2)

                print(f"Generated audio shape: {audio.shape if hasattr(audio, 'shape') else type(audio)}")

            except Exception as e:
                print(f"Generation error: {e}")
                import traceback
                traceback.print_exc()
                # Return silence as placeholder
                audio = torch.zeros(24000 * 2)

        # Ensure correct shape
        if isinstance(audio, torch.Tensor):
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            audio = audio.cpu().float()

        return audio

    def _decode_audio(self, sequences: torch.Tensor) -> torch.Tensor:
        """Decode token sequences to audio using the codec."""
        try:
            # Get base model (unwrap PEFT)
            base_model = self.model.base_model.model if hasattr(self.model, 'base_model') else self.model

            # Use codec to decode
            if hasattr(base_model, 'codec_model'):
                # Decode using Mimi codec
                with torch.no_grad():
                    audio = base_model.codec_model.decode(sequences)
                return audio
            else:
                print("No codec model found for decoding")
                return torch.zeros(24000 * 2)
        except Exception as e:
            print(f"Decode error: {e}")
            return torch.zeros(24000 * 2)

    def save_audio(self, audio: torch.Tensor, path: str, sample_rate: int = 24000):
        """Save audio tensor to file."""
        if audio.dtype == torch.float16:
            audio = audio.float()
        torchaudio.save(path, audio, sample_rate)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate speech with LoRA voice clone")
    parser.add_argument("--adapter", "-a", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--base-model", "-b", default="../models/csm-1b", help="Path to base CSM model")
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize")
    parser.add_argument("--output", "-o", default="output_lora.wav", help="Output audio file")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, mps, cpu)")

    args = parser.parse_args()

    # Create generator
    generator = LoRAVoiceGenerator(
        adapter_path=args.adapter,
        base_model_path=args.base_model,
        device=args.device,
    )

    # Generate
    audio = generator.generate(
        text=args.text,
        speaker=args.speaker,
        temperature=args.temperature,
    )

    # Save
    generator.save_audio(audio, args.output)

    print(f"\nGenerated: {args.output}")
    if audio.dim() >= 2:
        print(f"Duration: {audio.shape[1] / 24000:.2f} seconds")


if __name__ == "__main__":
    main()
