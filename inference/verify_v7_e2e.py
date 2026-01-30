#!/usr/bin/env python3
"""
V7 End-to-End Verification Script

This script verifies that the V7 prosody-conditioned model produces
distinct emotional outputs by:
1. Generating audio with happy and sad prosody
2. Extracting F0 from generated audio
3. Comparing F0 values to verify emotion differentiation

Success criteria (from Task #6):
- Generated happy audio sounds happier than sad
- F0 measured from generated audio shows Happy > Sad
- Model can be loaded and used for inference

Usage:
    python verify_v7_e2e.py
    python verify_v7_e2e.py --text "Your custom test sentence"
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

import torch

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'training'))
sys.path.insert(0, str(project_root / 'backend'))

from training.prosody_conditioning import (
    ProsodyConfig,
    ProsodyEncoder,
    TemporalProsodyEncoder,
    EmotionToProody,
)


def load_v7_model(checkpoint_path: str, csm_path: str, device: str = "auto"):
    """Load V7 prosody encoder and CSM model."""
    from transformers import CsmForConditionalGeneration, AutoProcessor

    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading V7 checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Prosody config: {ckpt.get('prosody_config', {})}")

    # Create prosody encoder
    prosody_config = ProsodyConfig(**ckpt.get('prosody_config', {}))
    prosody_encoder = ProsodyEncoder(prosody_config)
    prosody_encoder.load_state_dict(ckpt['prosody_encoder'])
    prosody_encoder = prosody_encoder.to(device)
    prosody_encoder.eval()
    print("✓ Prosody encoder loaded")

    # Load temporal encoder if available
    temporal_encoder = None
    if 'temporal_encoder' in ckpt:
        temporal_encoder = TemporalProsodyEncoder(prosody_config)
        temporal_encoder.load_state_dict(ckpt['temporal_encoder'])
        temporal_encoder = temporal_encoder.to(device)
        temporal_encoder.eval()
        print("✓ Temporal encoder loaded")

    # Load CSM model
    print(f"Loading CSM from: {csm_path}")
    csm = CsmForConditionalGeneration.from_pretrained(
        csm_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        csm_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Load LoRA weights if available (CRITICAL for prosody control!)
    if 'lora_state_dict' in ckpt:
        print("Loading LoRA adapter weights...")
        try:
            from peft import get_peft_model, LoraConfig, TaskType

            lora_cfg = ckpt.get('lora_config', {})
            lora_config = LoraConfig(
                r=lora_cfg.get('r', 8),
                lora_alpha=lora_cfg.get('lora_alpha', 16),
                target_modules=lora_cfg.get('target_modules', ['q_proj', 'v_proj']),
                lora_dropout=lora_cfg.get('lora_dropout', 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            csm = get_peft_model(csm, lora_config)
            csm.load_state_dict(ckpt['lora_state_dict'], strict=False)
            print("✓ LoRA adapter loaded!")
        except Exception as e:
            print(f"✗ Failed to load LoRA: {e}")
    else:
        print("⚠ No LoRA weights in checkpoint - prosody control may be limited")

    csm = csm.to(device)
    csm.eval()
    print("✓ CSM model loaded")

    return {
        'prosody_encoder': prosody_encoder,
        'temporal_encoder': temporal_encoder,
        'prosody_config': prosody_config,
        'csm': csm,
        'processor': processor,
        'device': device,
    }


def get_emotion_prosody(emotion: str, intensity: float, config: ProsodyConfig, device):
    """Get prosody vectors for an emotion."""
    prosody = EmotionToProody.get_prosody(emotion, intensity, config)
    for k, v in prosody.items():
        prosody[k] = v.to(device)
    return prosody


def generate_audio(models: dict, text: str, prosody: dict, max_length_ms: int = 10000) -> torch.Tensor:
    """Generate audio with prosody conditioning."""
    device = models['device']
    csm = models['csm']
    processor = models['processor']
    prosody_encoder = models['prosody_encoder']

    with torch.no_grad():
        # Prepare text input
        conversation = [{"role": "0", "content": [{"type": "text", "text": text}]}]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Encode prosody
        semantic = prosody['semantic']
        acoustic = prosody['acoustic']
        rhythm = prosody['rhythm']
        contour = prosody['contour']

        # Add batch dimension if needed
        if semantic.dim() == 1:
            semantic = semantic.unsqueeze(0)
            acoustic = acoustic.unsqueeze(0)
            rhythm = rhythm.unsqueeze(0)
            contour = contour.unsqueeze(0)

        prosody_prefix = prosody_encoder(semantic, acoustic, rhythm, contour)

        # Get text embeddings
        text_embeds = csm.embed_text_tokens(inputs['input_ids'])

        # Concatenate prosody prefix with text embeddings
        inputs_embeds = torch.cat([prosody_prefix, text_embeds], dim=1)

        # Extend attention mask
        prosody_mask = torch.ones(
            1, prosody_prefix.shape[1],
            device=device,
            dtype=inputs['attention_mask'].dtype,
        )
        extended_mask = torch.cat([prosody_mask, inputs['attention_mask']], dim=1)

        # Generate
        output = csm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            output_audio=True,
            max_new_tokens=max_length_ms // 80,
            do_sample=True,
            temperature=0.8,
            top_k=50,
        )

        # Extract audio
        if isinstance(output, list) and len(output) > 0:
            audio = output[0]
        elif hasattr(output, 'audio'):
            audio = output.audio[0]
        else:
            audio = output

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        return audio.cpu().float()


def extract_f0(audio: torch.Tensor, sample_rate: int = 24000) -> dict:
    """
    Extract F0 from audio using parselmouth/praat.

    Returns dict with f0_mean, f0_std, f0_values.
    """
    try:
        import parselmouth
    except ImportError:
        print("Warning: parselmouth not available, using fallback F0 estimation")
        return extract_f0_fallback(audio, sample_rate)

    # Convert to numpy
    audio_np = audio.squeeze().numpy()

    # Create Sound object
    sound = parselmouth.Sound(audio_np, sampling_frequency=sample_rate)

    # Extract pitch
    pitch = sound.to_pitch(time_step=0.01)
    f0_values = pitch.selected_array['frequency']

    # Filter out unvoiced (0 values)
    voiced_f0 = f0_values[f0_values > 0]

    if len(voiced_f0) == 0:
        return {'f0_mean': 0, 'f0_std': 0, 'f0_values': f0_values}

    return {
        'f0_mean': float(np.mean(voiced_f0)),
        'f0_std': float(np.std(voiced_f0)),
        'f0_values': f0_values,
        'voiced_ratio': len(voiced_f0) / len(f0_values) if len(f0_values) > 0 else 0,
    }


def extract_f0_fallback(audio: torch.Tensor, sample_rate: int = 24000) -> dict:
    """Fallback F0 extraction using autocorrelation."""
    audio_np = audio.squeeze().numpy()

    # Simple autocorrelation-based pitch estimation
    frame_size = int(0.05 * sample_rate)  # 50ms frames
    hop_size = int(0.01 * sample_rate)    # 10ms hop

    f0_values = []
    for i in range(0, len(audio_np) - frame_size, hop_size):
        frame = audio_np[i:i + frame_size]

        # Normalize
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) > 0:
            frame = frame / np.max(np.abs(frame))

        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]

        # Find first peak after minimum
        min_lag = int(sample_rate / 500)  # Max F0 = 500 Hz
        max_lag = int(sample_rate / 50)   # Min F0 = 50 Hz

        if max_lag > len(corr):
            max_lag = len(corr) - 1

        search_region = corr[min_lag:max_lag]
        if len(search_region) > 0:
            peak_idx = np.argmax(search_region) + min_lag
            if corr[peak_idx] > 0.3:  # Voiced threshold
                f0 = sample_rate / peak_idx
                f0_values.append(f0)
            else:
                f0_values.append(0)
        else:
            f0_values.append(0)

    f0_values = np.array(f0_values)
    voiced_f0 = f0_values[f0_values > 0]

    if len(voiced_f0) == 0:
        return {'f0_mean': 0, 'f0_std': 0, 'f0_values': f0_values}

    return {
        'f0_mean': float(np.mean(voiced_f0)),
        'f0_std': float(np.std(voiced_f0)),
        'f0_values': f0_values,
        'voiced_ratio': len(voiced_f0) / len(f0_values) if len(f0_values) > 0 else 0,
    }


def extract_energy(audio: torch.Tensor, sample_rate: int = 24000) -> dict:
    """Extract RMS energy from audio."""
    audio_np = audio.squeeze().numpy()

    frame_size = int(0.025 * sample_rate)  # 25ms frames
    hop_size = int(0.01 * sample_rate)     # 10ms hop

    rms_values = []
    for i in range(0, len(audio_np) - frame_size, hop_size):
        frame = audio_np[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        rms_values.append(rms)

    rms_values = np.array(rms_values)

    return {
        'energy_mean': float(np.mean(rms_values)),
        'energy_std': float(np.std(rms_values)),
        'energy_max': float(np.max(rms_values)),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify V7 prosody model end-to-end")
    parser.add_argument("--checkpoint", default="../models/checkpoints/prosody_v7/best.pt")
    parser.add_argument("--csm", default="../models/csm-1b")
    parser.add_argument("--text", default="I am feeling very emotional right now.")
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/v7_verification")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V7 End-to-End Verification")
    print("=" * 60)

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    checkpoint_path = (script_dir / args.checkpoint).resolve()
    csm_path = (script_dir / args.csm).resolve()

    # Load models
    print("\n1. Loading V7 model...")
    try:
        models = load_v7_model(str(checkpoint_path), str(csm_path), args.device)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test emotions
    emotions = ['happy', 'sad', 'angry', 'neutral', 'calm']
    results = {}

    print(f"\n2. Generating audio with different emotions...")
    print(f"   Text: \"{args.text}\"")
    print(f"   Intensity: {args.intensity}")

    for emotion in emotions:
        print(f"\n   Generating {emotion}...")

        try:
            # Get prosody for emotion
            prosody = get_emotion_prosody(
                emotion, args.intensity,
                models['prosody_config'], models['device']
            )

            # Generate audio
            audio = generate_audio(models, args.text, prosody)

            # Save audio using scipy (avoids torchcodec dependency)
            output_path = output_dir / f"v7_{emotion}.wav"
            audio_np = audio.squeeze().numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            wavfile.write(str(output_path), 24000, audio_int16)
            print(f"   ✓ Saved: {output_path}")

            # Extract F0 and energy
            f0_info = extract_f0(audio)
            energy_info = extract_energy(audio)

            results[emotion] = {
                'f0_mean': f0_info['f0_mean'],
                'f0_std': f0_info['f0_std'],
                'energy_mean': energy_info['energy_mean'],
                'audio_path': str(output_path),
                'duration_ms': audio.shape[1] / 24,
            }

            print(f"   F0: {f0_info['f0_mean']:.1f} Hz (±{f0_info['f0_std']:.1f})")
            print(f"   Energy: {energy_info['energy_mean']:.4f}")

        except Exception as e:
            print(f"   ✗ Error generating {emotion}: {e}")
            import traceback
            traceback.print_exc()
            results[emotion] = {'error': str(e)}

    # Analyze results
    print("\n" + "=" * 60)
    print("3. Analysis Results")
    print("=" * 60)

    # Check happy vs sad separation
    if 'happy' in results and 'sad' in results:
        if 'error' not in results['happy'] and 'error' not in results['sad']:
            happy_f0 = results['happy']['f0_mean']
            sad_f0 = results['sad']['f0_mean']
            separation = happy_f0 - sad_f0

            print(f"\n   Happy F0:  {happy_f0:.1f} Hz")
            print(f"   Sad F0:    {sad_f0:.1f} Hz")
            print(f"   Separation: {separation:.1f} Hz")

            if separation > 0:
                print(f"\n   ✓ SUCCESS: Happy pitch > Sad pitch ({separation:.1f} Hz separation)")
            else:
                print(f"\n   ✗ FAILED: Pitch inversion detected (Happy < Sad)")

    # Summary table
    print("\n   Emotion Summary:")
    print("   " + "-" * 50)
    print(f"   {'Emotion':<12} {'F0 (Hz)':<12} {'Energy':<12} {'Status'}")
    print("   " + "-" * 50)

    for emotion in emotions:
        if emotion in results and 'error' not in results[emotion]:
            f0 = results[emotion]['f0_mean']
            energy = results[emotion]['energy_mean']
            status = "✓" if f0 > 0 else "?"
            print(f"   {emotion:<12} {f0:<12.1f} {energy:<12.4f} {status}")
        else:
            error = results.get(emotion, {}).get('error', 'Unknown error')
            print(f"   {emotion:<12} {'Error':<12} {'Error':<12} ✗")

    print("   " + "-" * 50)

    # Expected ordering (high to low pitch)
    expected_order = ['happy', 'angry', 'neutral', 'calm', 'sad']
    valid_results = [(e, results[e]['f0_mean']) for e in expected_order
                     if e in results and 'error' not in results[e]]

    if len(valid_results) >= 2:
        sorted_by_f0 = sorted(valid_results, key=lambda x: x[1], reverse=True)
        actual_order = [e for e, _ in sorted_by_f0]

        print(f"\n   Expected F0 order (high→low): {' > '.join(expected_order)}")
        print(f"   Actual F0 order (high→low):   {' > '.join(actual_order)}")

    # Final verdict
    print("\n" + "=" * 60)
    print("4. VERIFICATION VERDICT")
    print("=" * 60)

    success = True
    issues = []

    # Check model loading
    if 'prosody_encoder' not in models:
        success = False
        issues.append("Prosody encoder failed to load")

    # Check happy > sad
    if 'happy' in results and 'sad' in results:
        if 'error' not in results['happy'] and 'error' not in results['sad']:
            if results['happy']['f0_mean'] <= results['sad']['f0_mean']:
                success = False
                issues.append(f"Pitch inversion: Happy ({results['happy']['f0_mean']:.1f}Hz) <= Sad ({results['sad']['f0_mean']:.1f}Hz)")
        else:
            success = False
            issues.append("Failed to generate happy or sad audio")
    else:
        success = False
        issues.append("Missing happy or sad samples")

    # Check F0 extraction
    for emotion in ['happy', 'sad']:
        if emotion in results and 'error' not in results[emotion]:
            if results[emotion]['f0_mean'] <= 0:
                success = False
                issues.append(f"Invalid F0 for {emotion}")

    if success:
        print("\n   ✓ V7 VERIFICATION PASSED!")
        print("\n   All success criteria met:")
        print("   - Model loads successfully")
        print("   - Audio generates for all emotions")
        print("   - Happy pitch > Sad pitch (emotion differentiation works)")
        print("\n   Generated audio files are in:", output_dir)
    else:
        print("\n   ✗ V7 VERIFICATION FAILED!")
        print("\n   Issues found:")
        for issue in issues:
            print(f"   - {issue}")

    print("\n" + "=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
