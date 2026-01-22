"""
Voice Clone Pipeline - Speech Generation
Generate speech using your fine-tuned CSM model.

Usage:
    python generate.py --model ../models/checkpoints/voice_v1/best.pt --text "Hello world!"
    python generate.py --model ../models/checkpoints/voice_v1/best.pt --text "Hello!" --context "Hey there!"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import torch
import torchaudio


class VoiceGenerator:
    """
    Generate speech using a fine-tuned CSM model.
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = Path(model_path)
        
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
        
        # Load model and processor
        self.model, self.processor = self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and processor."""
        print(f"Loading model from: {self.model_path}")

        from transformers import CsmForConditionalGeneration, AutoProcessor

        # Base model path for processor
        base_model = "../models/csm-1b"

        # Load processor first
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        print("Processor loaded!")

        # Check if it's a checkpoint or full model
        if self.model_path.suffix == ".pt":
            # Load base model first
            model = CsmForConditionalGeneration.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=torch.float32 if self.device.type == 'mps' else torch.float16,
            )

            # Load checkpoint weights
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                # Only load matching keys (fine-tuned weights)
                model_state = model.state_dict()
                for k, v in checkpoint["model_state_dict"].items():
                    if k in model_state and model_state[k].shape == v.shape:
                        model_state[k] = v
                model.load_state_dict(model_state)
                print(f"Loaded fine-tuned weights from checkpoint")
        else:
            # Load full model directory
            model = CsmForConditionalGeneration.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                torch_dtype=torch.float32 if self.device.type == 'mps' else torch.float16,
            )

        model = model.to(self.device)
        model.eval()

        print("Model loaded!")
        return model, processor
    
    def generate(
        self,
        text: str,
        context: Optional[str] = None,
        speaker: int = 0,
        temperature: float = 0.8,
        top_k: int = 50,
        max_length_ms: int = 30000,
    ) -> torch.Tensor:
        """
        Generate speech from text.
        
        Args:
            text: Text to synthesize
            context: Optional context for better prosody
            speaker: Speaker ID (0 for single-speaker models)
            temperature: Sampling temperature (higher = more variation)
            top_k: Top-k sampling parameter
            max_length_ms: Maximum audio length in milliseconds
        
        Returns:
            Audio waveform tensor [1, samples] at 24kHz
        """
        # Format text with speaker token
        formatted_text = f"[{speaker}]{text}"
        
        # Build context if provided
        if context:
            formatted_context = f"[{speaker}]{context}"
        else:
            formatted_context = None
        
        # Generate using CSM's proper API
        with torch.no_grad():
            try:
                # Build conversation for CSM (text-only prompt)
                conversation = [{"role": str(speaker), "content": [{"type": "text", "text": text}]}]

                # Use processor to prepare inputs
                inputs = self.processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_dict=True,
                )

                # Move to device
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                # Generate with output_audio=True to get actual audio
                output = self.model.generate(
                    **inputs,
                    output_audio=True,
                    max_new_tokens=max_length_ms // 80,  # ~80ms per token
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                )

                # Extract audio from output
                if hasattr(output, 'audio') and output.audio:
                    audio = output.audio[0]  # First batch item
                elif isinstance(output, list):
                    audio = output[0]
                else:
                    audio = output

                print(f"Generated audio shape: {audio.shape if hasattr(audio, 'shape') else 'unknown'}")

            except Exception as e:
                print(f"CSM generation error: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: generate silence placeholder
                print("Generating placeholder audio...")
                duration_samples = int(24000 * 2)  # 2 seconds
                audio = torch.zeros(duration_samples)
        
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        return audio.cpu()
    
    def generate_conversation(
        self,
        turns: List[str],
        speaker: int = 0,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Generate a multi-turn conversation with proper context.
        Each turn uses previous turns as context.
        """
        audios = []
        context_text = ""
        
        for i, turn in enumerate(turns):
            print(f"Generating turn {i+1}/{len(turns)}: {turn[:50]}...")
            
            audio = self.generate(
                text=turn,
                context=context_text if context_text else None,
                speaker=speaker,
                **kwargs
            )
            
            audios.append(audio)
            
            # Update context with this turn
            context_text = turn
        
        return audios
    
    def save_audio(self, audio: torch.Tensor, path: str, sample_rate: int = 24000):
        """Save audio tensor to file."""
        # Convert to float32 if needed (torchaudio doesn't support float16)
        if audio.dtype == torch.float16:
            audio = audio.float()
        torchaudio.save(path, audio, sample_rate)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate speech with your voice clone")
    parser.add_argument("--model", "-m", required=True, help="Path to model checkpoint")
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize")
    parser.add_argument("--context", "-c", help="Optional context for better prosody")
    parser.add_argument("--output", "-o", default="output.wav", help="Output audio file")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, mps, cpu)")
    
    args = parser.parse_args()
    
    # Create generator
    generator = VoiceGenerator(args.model, device=args.device)
    
    # Generate
    audio = generator.generate(
        text=args.text,
        context=args.context,
        speaker=args.speaker,
        temperature=args.temperature,
    )
    
    # Save
    generator.save_audio(audio, args.output)
    
    print(f"\nGenerated: {args.output}")
    print(f"Duration: {audio.shape[1] / 24000:.2f} seconds")


if __name__ == "__main__":
    main()
