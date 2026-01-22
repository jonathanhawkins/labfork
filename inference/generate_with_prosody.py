"""
Controllable Voice Generation with Prosody Conditioning

Generate speech with emotion and style control using the prosody conditioning module.

Usage:
    # Generate with emotion control
    python generate_with_prosody.py --text "I'm so happy to see you!" --emotion happy

    # Generate with reference audio (style transfer)
    python generate_with_prosody.py --text "Hello world" --reference ref_audio.wav

    # Generate with custom prosody
    python generate_with_prosody.py --text "Testing" --pitch high --energy 0.8 --rate fast

    # Generate with keyframe prosody control (emotion trajectory over time)
    python generate_with_prosody.py --text "Hello, I am feeling great today!" \\
        --keyframes '[{"time":0,"emotion":"neutral","intensity":0.5},{"time":1,"emotion":"happy","intensity":0.9}]' \\
        --duration 3.0

Examples:
    # Happy greeting
    python generate_with_prosody.py -t "Good morning everyone!" -e happy -o happy_greeting.wav

    # Calm announcement
    python generate_with_prosody.py -t "The meeting will begin shortly." -e calm -o announcement.wav

    # Style transfer from reference
    python generate_with_prosody.py -t "This is my cloned voice." -r reference.wav -o cloned.wav

    # Keyframe trajectory: neutral -> happy -> calm
    python generate_with_prosody.py -t "I woke up feeling okay, then got excited, and now I'm peaceful." \\
        --keyframes '[{"time":0,"emotion":"neutral","intensity":0.5},{"time":0.4,"emotion":"happy","intensity":0.9},{"time":1,"emotion":"calm","intensity":0.7}]' \\
        --duration 5.0 -o trajectory.wav
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from training.prosody_conditioning import (
    ProsodyConfig,
    ProsodyEncoder,
    TemporalProsodyEncoder,
    EmotionToProody,
    extract_prosody_for_conditioning,
)
from keyframe_prosody import ProsodyKeyframe, keyframes_to_prosody, get_temporal_prosody_tokens


class ControllableVoiceGenerator:
    """
    Generate speech with prosody control.

    This is the innovation: prosody analysis → controllable generation.
    """

    def __init__(
        self,
        csm_path: str,
        prosody_checkpoint: Optional[str] = None,
        lora_adapter: Optional[str] = None,
        device: str = "auto",
    ):
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

        # Load models
        self.csm, self.processor = self._load_csm(csm_path, lora_adapter)
        self.prosody_encoder = self._load_prosody_encoder(prosody_checkpoint)
        self.prosody_config = ProsodyConfig()

        # Try to load prosody analyzer for reference audio
        try:
            from prosody_analyzer import CompleteProsodyAnalyzer
            self.analyzer = CompleteProsodyAnalyzer(use_qwen=False)
            print("Prosody analyzer loaded - reference audio supported")
        except Exception:
            self.analyzer = None
            print("Prosody analyzer not available - using emotion presets only")

    def _load_csm(self, csm_path: str, lora_adapter: Optional[str]):
        """Load CSM model with optional LoRA."""
        from transformers import CsmForConditionalGeneration, AutoProcessor

        # Resolve path
        csm_path = Path(csm_path).resolve()
        print(f"Loading CSM from: {csm_path}")

        model = CsmForConditionalGeneration.from_pretrained(
            str(csm_path),
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
        )

        processor = AutoProcessor.from_pretrained(
            str(csm_path),
            trust_remote_code=True,
            local_files_only=True,
        )

        # Apply LoRA if provided
        if lora_adapter:
            try:
                from peft import PeftModel
                lora_path = Path(lora_adapter).resolve()
                print(f"Loading LoRA adapter from: {lora_path}")
                model = PeftModel.from_pretrained(
                    model,
                    str(lora_path),
                    is_trainable=False,
                )
                print("LoRA adapter loaded")
            except Exception as e:
                print(f"Could not load LoRA adapter: {e}")

        model = model.to(self.device)
        model.eval()

        return model, processor

    def _load_prosody_encoder(self, checkpoint: Optional[str]) -> Optional[ProsodyEncoder]:
        """Load trained prosody encoder and create temporal encoder."""
        self.temporal_encoder = None

        if not checkpoint:
            print("No prosody encoder checkpoint - using emotion presets")
            # Still create temporal encoder for keyframe support
            self._create_temporal_encoder()
            return None

        try:
            ckpt = torch.load(checkpoint, map_location=self.device)

            config = ProsodyConfig(**ckpt.get('prosody_config', {}))
            self.prosody_config = config
            encoder = ProsodyEncoder(config)
            encoder.load_state_dict(ckpt['prosody_encoder'])
            encoder = encoder.to(self.device)
            encoder.eval()

            print(f"Loaded prosody encoder from: {checkpoint}")

            # Also load or create temporal encoder for keyframe support
            if 'temporal_encoder' in ckpt:
                self.temporal_encoder = TemporalProsodyEncoder(config)
                self.temporal_encoder.load_state_dict(ckpt['temporal_encoder'])
                self.temporal_encoder = self.temporal_encoder.to(self.device)
                self.temporal_encoder.eval()
                print("Loaded temporal encoder from checkpoint")
            else:
                self._create_temporal_encoder(global_encoder=encoder)

            return encoder

        except Exception as e:
            print(f"Could not load prosody encoder: {e}")
            return None

    def _create_temporal_encoder(self, global_encoder: Optional[ProsodyEncoder] = None):
        """Create a fresh temporal encoder for keyframe support."""
        try:
            self.temporal_encoder = TemporalProsodyEncoder(self.prosody_config)
            self.temporal_encoder = self.temporal_encoder.to(self.device)
            self.temporal_encoder.eval()
            print("Created temporal encoder for keyframe support")
            if global_encoder is not None:
                self._init_temporal_from_global(global_encoder)
        except Exception as e:
            print(f"Could not create temporal encoder: {e}")
            self.temporal_encoder = None

    def _init_temporal_from_global(self, global_encoder: ProsodyEncoder) -> None:
        """Initialize temporal encoder weights from the global prosody encoder."""
        if self.temporal_encoder is None:
            return
        if hasattr(self.temporal_encoder, "init_from_global_encoder"):
            self.temporal_encoder.init_from_global_encoder(global_encoder)
            print("Initialized temporal encoder from global prosody encoder")

    def _collapse_temporal_to_global(
        self,
        prosody: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Collapse temporal prosody tokens into a single global vector per type."""
        collapsed = {}
        for key in ["semantic", "acoustic", "rhythm", "contour"]:
            tensor = prosody[key]
            if tensor.dim() == 2:
                collapsed[key] = tensor.mean(dim=0, keepdim=True)
            elif tensor.dim() == 3:
                collapsed[key] = tensor.mean(dim=1)
            else:
                collapsed[key] = tensor
        return collapsed

    def extract_prosody_from_reference(
        self,
        audio_path: str,
    ) -> Dict[str, torch.Tensor]:
        """Extract prosody from reference audio for style transfer."""
        if self.analyzer is None:
            raise RuntimeError("Prosody analyzer not available")

        # Analyze prosody - CompleteProsodyAnalyzer.analyze() expects a file path
        prosody_result = self.analyzer.analyze(audio_path)
        prosody_dict = prosody_result.to_dict()

        # Convert to conditioning format
        prosody = extract_prosody_for_conditioning(prosody_dict, self.prosody_config)

        # Move to device
        for k, v in prosody.items():
            prosody[k] = v.to(self.device)

        return prosody

    def get_emotion_prosody(
        self,
        emotion: str,
        intensity: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Get prosody vectors for an emotion."""
        prosody = EmotionToProody.get_prosody(emotion, intensity, self.prosody_config)

        # Move to device
        for k, v in prosody.items():
            prosody[k] = v.to(self.device)

        return prosody

    def get_custom_prosody(
        self,
        pitch: str = "medium",      # low, medium, high
        energy: float = 0.5,        # 0.0 to 1.0
        rate: str = "medium",       # slow, medium, fast
        emotion_mix: Optional[Dict[str, float]] = None,  # {"happy": 0.5, "calm": 0.5}
    ) -> Dict[str, torch.Tensor]:
        """Create custom prosody from parameters."""
        config = self.prosody_config

        # Map pitch
        pitch_values = {"low": 0.3, "medium": 0.5, "high": 0.7}
        pitch_val = pitch_values.get(pitch, 0.5)

        # Map rate
        rate_values = {"slow": 0.3, "medium": 0.5, "fast": 0.7}
        rate_val = rate_values.get(rate, 0.5)

        # Create semantic (emotion mix)
        semantic = torch.zeros(config.semantic_dim)
        if emotion_mix:
            emotions = list(EmotionToProody.EMOTION_PROFILES.keys())
            for emotion, weight in emotion_mix.items():
                if emotion in emotions:
                    idx = emotions.index(emotion)
                    if idx < config.semantic_dim:
                        semantic[idx] = weight

        # Create acoustic
        acoustic = torch.zeros(config.acoustic_dim)
        acoustic[0] = pitch_val      # pitch mean
        acoustic[1] = 0.3            # pitch std
        acoustic[2] = energy         # energy
        acoustic[3:] = 0.5           # defaults

        # Create rhythm
        rhythm = torch.zeros(config.rhythm_dim)
        rhythm[0] = rate_val         # speaking rate
        rhythm[1] = 0.3 if rate_val > 0.5 else 0.5  # pause ratio
        rhythm[2:] = 0.5

        # Create contour (neutral flat)
        contour = torch.ones(config.contour_dim) * 0.5

        prosody = {
            "semantic": semantic.unsqueeze(0).to(self.device),
            "acoustic": acoustic.unsqueeze(0).to(self.device),
            "rhythm": rhythm.unsqueeze(0).to(self.device),
            "contour": contour.unsqueeze(0).to(self.device),
        }

        return prosody

    def get_keyframe_prosody(
        self,
        keyframes_json: str,
        duration_seconds: float = 5.0,
        use_temporal: bool = True,
        num_segments: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Get prosody from keyframe JSON with temporal/per-segment control.

        This method preserves keyframe edits by converting them to temporal
        tokens instead of averaging into global tokens.

        Args:
            keyframes_json: JSON array of keyframes, e.g.:
                '[{"time":0,"emotion":"neutral","intensity":0.5},
                  {"time":1,"emotion":"happy","intensity":0.9}]'
            duration_seconds: Total duration for interpreting keyframe times.
            use_temporal: If True, return temporal tokens that preserve keyframe
                         edits. If False, return averaged global tokens.
            num_segments: Number of time segments for temporal conditioning.
                         Each segment gets one prosody token. Default 4 matches
                         the model's num_prosody_tokens setting.

        Returns:
            Prosody conditioning dict with tensors on device.
            If use_temporal=True, tensors have shape [num_segments, dim] and
            include '_is_temporal': True flag.
            If use_temporal=False, tensors have shape [1, dim] (averaged).
        """
        import json

        max_segments = self.prosody_config.num_prosody_tokens * 4
        if num_segments > max_segments:
            raise ValueError(
                f"num_segments ({num_segments}) exceeds max supported ({max_segments}). "
                "Increase ProsodyConfig.num_prosody_tokens or reduce --segments."
            )

        keyframes_data = json.loads(keyframes_json)
        keyframes = [ProsodyKeyframe(**kf) for kf in keyframes_data]

        # Get dense prosody from keyframes
        prosody = keyframes_to_prosody(
            keyframes,
            duration_seconds=duration_seconds,
            config=self.prosody_config,
        )

        if use_temporal:
            # Convert dense prosody to temporal tokens (preserves keyframe edits)
            temporal = get_temporal_prosody_tokens(prosody, num_segments=num_segments)
            temporal['_is_temporal'] = True
            temporal['_num_segments'] = num_segments

            # Move to device
            for k, v in temporal.items():
                if isinstance(v, torch.Tensor):
                    temporal[k] = v.to(self.device)

            return temporal
        else:
            # Use averaged global tokens (original behavior, loses keyframe edits)
            prosody['_is_temporal'] = False

            # Move to device
            for k, v in prosody.items():
                if isinstance(v, torch.Tensor):
                    prosody[k] = v.to(self.device)

            return prosody

    def generate(
        self,
        text: str,
        prosody: Dict[str, torch.Tensor],
        speaker: int = 0,
        temperature: float = 0.8,
        top_k: int = 50,
        max_audio_length_ms: int = 10000,
    ) -> torch.Tensor:
        """
        Generate speech with prosody conditioning.

        Supports both global and temporal prosody conditioning:
        - Global: prosody with shape [1, dim] - single style for entire utterance
        - Temporal: prosody with shape [num_segments, dim] - per-segment control
                   (set by get_keyframe_prosody with use_temporal=True)

        Args:
            text: Text to synthesize
            prosody: Prosody conditioning dict. If '_is_temporal' key is True,
                    uses temporal conditioning that preserves keyframe edits.
            speaker: Speaker ID
            temperature: Sampling temperature
            top_k: Top-k sampling
            max_audio_length_ms: Maximum audio length

        Returns:
            Audio tensor
        """
        is_temporal = prosody.get('_is_temporal', False)
        num_segments = prosody.get('_num_segments', self.prosody_config.num_prosody_tokens)

        print(f"Generating: '{text}'")
        if is_temporal:
            print(f"Prosody mode: TEMPORAL ({num_segments} segments, preserves keyframe edits)")
            print(f"Prosody - semantic range: [{prosody['semantic'].min():.2f}, {prosody['semantic'].max():.2f}], "
                  f"acoustic range: [{prosody['acoustic'].min():.2f}, {prosody['acoustic'].max():.2f}]")
        else:
            print(f"Prosody mode: GLOBAL (averaged, loses temporal variation)")
            print(f"Prosody - semantic: {prosody['semantic'].mean():.2f}, "
                  f"acoustic: {prosody['acoustic'].mean():.2f}, "
                  f"rhythm: {prosody['rhythm'].mean():.2f}")

        with torch.no_grad():
            # Prepare text input
            conversation = [
                {"role": str(speaker), "content": [{"type": "text", "text": text}]}
            ]

            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
            )
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            # If we have a trained prosody encoder, use it
            if self.prosody_encoder is not None:
                if is_temporal and self.temporal_encoder is None:
                    print("Temporal encoder unavailable; falling back to global prosody averaging.")
                    prosody = self._collapse_temporal_to_global(prosody)
                    is_temporal = False

                if is_temporal and hasattr(self, 'temporal_encoder') and self.temporal_encoder is not None:
                    # TEMPORAL MODE: Use temporal encoder for per-segment control
                    # Add batch dimension if needed
                    semantic = prosody['semantic']
                    acoustic = prosody['acoustic']
                    rhythm = prosody['rhythm']
                    contour = prosody['contour']

                    if semantic.dim() == 2:
                        semantic = semantic.unsqueeze(0)
                        acoustic = acoustic.unsqueeze(0)
                        rhythm = rhythm.unsqueeze(0)
                        contour = contour.unsqueeze(0)

                    # Encode temporal prosody
                    prosody_prefix = self.temporal_encoder(
                        semantic, acoustic, rhythm, contour
                    )
                else:
                    # GLOBAL MODE: Use standard encoder (original behavior)
                    # Add batch dimension if needed
                    semantic = prosody['semantic']
                    acoustic = prosody['acoustic']
                    rhythm = prosody['rhythm']
                    contour = prosody['contour']

                    if semantic.dim() == 1:
                        semantic = semantic.unsqueeze(0)
                        acoustic = acoustic.unsqueeze(0)
                        rhythm = rhythm.unsqueeze(0)
                        contour = contour.unsqueeze(0)

                    prosody_prefix = self.prosody_encoder(
                        semantic, acoustic, rhythm, contour
                    )

                # Get text embeddings
                text_embeds = self.csm.embed_text_tokens(inputs['input_ids'])

                # Concatenate prosody prefix
                inputs_embeds = torch.cat([prosody_prefix, text_embeds], dim=1)

                # Extend attention mask
                prosody_mask = torch.ones(
                    1, prosody_prefix.shape[1],
                    device=self.device,
                    dtype=inputs['attention_mask'].dtype,
                )
                extended_mask = torch.cat([prosody_mask, inputs['attention_mask']], dim=1)

                # Generate with prefix
                output = self.csm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=extended_mask,
                    output_audio=True,
                    max_new_tokens=max_audio_length_ms // 80,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                )
            else:
                # Without trained encoder, just generate normally
                # (prosody info is informational only)
                if is_temporal:
                    print("Warning: temporal prosody requested but no prosody encoder is loaded. "
                          "Keyframe edits will be ignored.")
                output = self.csm.generate(
                    **inputs,
                    output_audio=True,
                    max_new_tokens=max_audio_length_ms // 80,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                )

            # Extract audio
            if isinstance(output, list) and len(output) > 0:
                audio = output[0]
            elif hasattr(output, 'audio'):
                audio = output.audio[0]
            else:
                audio = output

            print(f"Generated audio shape: {audio.shape}")

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        return audio.cpu().float()

    def save_audio(self, audio: torch.Tensor, path: str, sample_rate: int = 24000):
        """Save audio to file."""
        torchaudio.save(path, audio, sample_rate)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate speech with prosody control")

    # Required
    parser.add_argument("-t", "--text", required=True, help="Text to synthesize")

    # Model paths
    parser.add_argument("--csm", default="../models/csm-1b", help="CSM model path")
    parser.add_argument("--lora", help="LoRA adapter path")
    parser.add_argument("--prosody-ckpt", help="Trained prosody encoder checkpoint")

    # Prosody control
    parser.add_argument("-e", "--emotion", help="Emotion (happy, sad, angry, calm, surprised)")
    parser.add_argument("-r", "--reference", help="Reference audio for style transfer")
    parser.add_argument("--pitch", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--energy", type=float, default=0.5, help="Energy level 0-1")
    parser.add_argument("--rate", choices=["slow", "medium", "fast"], default="medium")
    parser.add_argument("--intensity", type=float, default=1.0, help="Emotion intensity 0-1")
    parser.add_argument(
        "--keyframes",
        help='Keyframes as JSON array, e.g. \'[{"time":0,"emotion":"neutral","intensity":0.5},{"time":1,"emotion":"happy","intensity":0.9}]\''
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds for keyframe interpolation")
    parser.add_argument(
        "--segments", type=int, default=4,
        help="Number of temporal segments for keyframe control (default: 4)"
    )
    parser.add_argument(
        "--no-temporal", action="store_true",
        help="Use global (averaged) prosody instead of temporal for keyframes. "
             "WARNING: This loses per-keyframe edits!"
    )

    # Generation
    parser.add_argument("-o", "--output", default="output_prosody.wav", help="Output file")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    # Create generator
    generator = ControllableVoiceGenerator(
        csm_path=args.csm,
        lora_adapter=args.lora,
        prosody_checkpoint=args.prosody_ckpt,
        device=args.device,
    )

    # Get prosody
    if args.reference:
        print(f"\nExtracting prosody from reference: {args.reference}")
        prosody = generator.extract_prosody_from_reference(args.reference)
        print("Style transfer mode: matching reference prosody")
    elif args.keyframes:
        print(f"\nUsing keyframes: {args.keyframes}")
        use_temporal = not args.no_temporal
        prosody = generator.get_keyframe_prosody(
            args.keyframes,
            duration_seconds=args.duration,
            use_temporal=use_temporal,
            num_segments=args.segments,
        )
        if use_temporal:
            print(f"Keyframe mode: TEMPORAL ({args.segments} segments, preserves keyframe edits)")
        else:
            print(f"Keyframe mode: GLOBAL (averaged, loses temporal variation)")
    elif args.emotion:
        print(f"\nUsing emotion preset: {args.emotion} (intensity={args.intensity})")
        prosody = generator.get_emotion_prosody(args.emotion, args.intensity)
    else:
        print(f"\nUsing custom prosody: pitch={args.pitch}, energy={args.energy}, rate={args.rate}")
        prosody = generator.get_custom_prosody(
            pitch=args.pitch,
            energy=args.energy,
            rate=args.rate,
        )

    # Generate
    audio = generator.generate(
        text=args.text,
        prosody=prosody,
        temperature=args.temperature,
    )

    # Save
    generator.save_audio(audio, args.output)

    print(f"\nGenerated: {args.output}")
    print(f"Duration: {audio.shape[1] / 24000:.2f} seconds")


if __name__ == "__main__":
    main()
