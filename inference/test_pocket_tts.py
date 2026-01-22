#!/usr/bin/env python3
"""
Test Pocket TTS voice cloning with recorded voice samples.

Pocket TTS uses zero-shot voice cloning - no fine-tuning needed.
Just provide a reference audio file and it clones the voice at inference time.

Usage:
    # List available voice samples
    python test_pocket_tts.py --list-samples

    # Clone voice with default text
    python test_pocket_tts.py -r path/to/voice.wav

    # Clone voice with custom text
    python test_pocket_tts.py -r path/to/voice.wav -t "Your custom text here"

    # Generate samples for all emotions
    python test_pocket_tts.py --all-emotions

    # Use built-in voice (no voice cloning needed)
    python test_pocket_tts.py --builtin alba
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_voice_samples(data_dir: Path) -> list[Path]:
    """Find all recorded voice samples."""
    samples = []
    for wav_file in data_dir.rglob("*.wav"):
        # Skip raw files, prefer processed ones
        if "_raw" not in wav_file.name:
            samples.append(wav_file)
    return sorted(samples)


def load_model():
    """Load Pocket TTS model."""
    from pocket_tts import TTSModel

    print("Loading Pocket TTS model...")
    model = TTSModel.load_model()
    print(f"  Voice cloning enabled: {model.has_voice_cloning}")
    print(f"  Sample rate: {model.sample_rate} Hz")
    return model


def save_audio(audio, output_path: Path, sample_rate: int = 24000):
    """Save audio tensor to WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_np = audio.cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()

    # Normalize to int16 range
    audio_np = (audio_np * 32767).astype(np.int16)
    wav.write(str(output_path), sample_rate, audio_np)


def generate_with_voice_cloning(
    model,
    reference_audio: Path,
    text: str,
    output_path: Path,
):
    """Generate speech using voice cloning from reference audio."""
    if not model.has_voice_cloning:
        print("ERROR: Voice cloning not enabled.")
        print("Please accept terms at: https://huggingface.co/kyutai/pocket-tts")
        sys.exit(1)

    print(f"\nReference: {reference_audio.name}")
    print(f"Text: {text}")

    voice_state = model.get_state_for_audio_prompt(str(reference_audio))
    audio = model.generate_audio(voice_state, text)

    save_audio(audio, output_path, model.sample_rate)
    print(f"Saved: {output_path}")
    return output_path


def generate_with_builtin_voice(
    model,
    voice_name: str,
    text: str,
    output_path: Path,
):
    """Generate speech using a built-in voice."""
    builtin_voices = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']

    if voice_name not in builtin_voices:
        print(f"ERROR: Unknown voice '{voice_name}'")
        print(f"Available voices: {', '.join(builtin_voices)}")
        sys.exit(1)

    print(f"\nVoice: {voice_name}")
    print(f"Text: {text}")

    voice_state = model.get_state_for_audio_prompt(voice_name)
    audio = model.generate_audio(voice_state, text)

    save_audio(audio, output_path, model.sample_rate)
    print(f"Saved: {output_path}")
    return output_path


def generate_all_emotions(model, data_dir: Path, output_dir: Path):
    """Generate samples for all emotional variations."""
    emotion_samples = {
        'calm': ('calm_1_111759_00de.wav', 'Hello, this is my calm and relaxed voice.'),
        'happy': ('happy_1_111944_2a14.wav', 'This is my happy voice! I am so excited about this.'),
        'angry': ('angry_1_112142_b016.wav', 'This is my angry voice. I am not pleased with this.'),
        'sad': ('sad_1_112036_01e3.wav', 'This is my sad voice. Things are not going well.'),
        'fearful': ('fearful_1_112312_6ee2.wav', 'This is my fearful voice. I am worried about this.'),
        'surprised': ('surprised_1_112425_4def.wav', 'Oh wow! This is my surprised voice!'),
        'excited': ('excited_1_112540_298c.wav', 'This is amazing! I am so excited about this technology!'),
    }

    # Find the session directory
    session_dirs = list(data_dir.glob("20*"))
    if not session_dirs:
        print("No voice sample sessions found.")
        return

    session_dir = session_dirs[-1]  # Use most recent
    print(f"Using session: {session_dir.name}")

    generated = []
    for emotion, (filename, text) in emotion_samples.items():
        voice_path = session_dir / filename
        if not voice_path.exists():
            # Try to find any file with this emotion
            matches = list(session_dir.glob(f"{emotion}_*.wav"))
            matches = [m for m in matches if "_raw" not in m.name]
            if matches:
                voice_path = matches[0]
            else:
                print(f"Skipping {emotion} - no sample found")
                continue

        output_path = output_dir / f"pocket_tts_{emotion}.wav"
        generate_with_voice_cloning(model, voice_path, text, output_path)
        generated.append(output_path)

    print(f"\n=== Generated {len(generated)} samples ===")
    for f in generated:
        print(f"  {f}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Pocket TTS voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--reference", "-r",
        type=Path,
        help="Path to reference audio file for voice cloning"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default="Hello, this is a test of voice cloning with Pocket TTS. The quick brown fox jumps over the lazy dog.",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for generated audio"
    )
    parser.add_argument(
        "--list-samples", "-l",
        action="store_true",
        help="List available voice samples"
    )
    parser.add_argument(
        "--all-emotions",
        action="store_true",
        help="Generate samples for all emotional variations"
    )
    parser.add_argument(
        "--builtin",
        type=str,
        help="Use a built-in voice (alba, marius, javert, jean, fantine, cosette, eponine, azelma)"
    )

    args = parser.parse_args()

    data_dir = project_root / "data" / "voice_samples"
    output_dir = project_root / "inference" / "outputs"

    if args.list_samples:
        samples = find_voice_samples(data_dir)
        print(f"Found {len(samples)} voice samples:")
        for i, sample in enumerate(samples[:30]):
            print(f"  {i+1}. {sample.relative_to(project_root)}")
        if len(samples) > 30:
            print(f"  ... and {len(samples) - 30} more")
        return

    model = load_model()

    if args.all_emotions:
        generate_all_emotions(model, data_dir, output_dir)
        return

    if args.builtin:
        output_path = args.output or output_dir / f"pocket_tts_{args.builtin}.wav"
        generate_with_builtin_voice(model, args.builtin, args.text, output_path)
        return

    # Voice cloning mode
    if args.reference is None:
        samples = find_voice_samples(data_dir)
        if not samples:
            print("No voice samples found. Please record some samples first.")
            print("Or use --builtin <voice> for built-in voices.")
            sys.exit(1)
        args.reference = samples[0]
        print(f"Using first available sample: {args.reference}")

    if not args.reference.exists():
        print(f"Reference audio not found: {args.reference}")
        sys.exit(1)

    output_path = args.output or output_dir / "pocket_tts_cloned.wav"
    generate_with_voice_cloning(model, args.reference, args.text, output_path)


if __name__ == "__main__":
    main()
