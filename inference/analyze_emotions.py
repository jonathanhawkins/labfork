#!/usr/bin/env python3
"""Analyze emotion samples for pitch and intensity."""
import numpy as np
import parselmouth
import sys
from pathlib import Path

def analyze(path):
    sound = parselmouth.Sound(str(path))
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]
    intensity = sound.to_intensity()
    return np.mean(pitch_values), np.mean(intensity.values.flatten())

def main():
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/prosody_v4")

    print("=" * 60)
    print("PROSODY ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Emotion     Pitch (Hz)      Intensity (dB)")
    print("-" * 50)

    results = {}
    for e in ["neutral", "happy", "sad", "angry"]:
        path = base_dir / f"{e}_test.wav"
        if path.exists():
            p, i = analyze(path)
            results[e] = {"pitch": p, "intensity": i}
            print(f"{e:<12}{p:>8.1f}         {i:>8.1f}")
        else:
            print(f"{e:<12}(not found)")

    print("\n" + "=" * 60)
    print("EXPECTED PATTERNS:")
    print("=" * 60)
    print("Happy:   HIGHEST pitch, HIGH energy")
    print("Angry:   HIGH pitch, HIGHEST energy")
    print("Neutral: MEDIUM pitch, MEDIUM energy")
    print("Sad:     LOWEST pitch, LOW energy")

    print("\n" + "=" * 60)
    print("PATTERN CHECK:")
    print("=" * 60)

    if len(results) >= 3:
        pitches = {e: results[e]["pitch"] for e in results}
        sorted_pitch = sorted(pitches.items(), key=lambda x: x[1])
        print(f"Pitch order (low->high): {' < '.join([e[0] for e in sorted_pitch])}")

        # Check patterns
        happy_highest = pitches.get("happy", 0) >= max(pitches.get("sad", 0), pitches.get("neutral", 0))
        sad_lowest = pitches.get("sad", float("inf")) <= min(pitches.get("happy", float("inf")), pitches.get("neutral", float("inf")))

        print(f"Happy has highest pitch: {'YES' if happy_highest else 'NO'}")
        print(f"Sad has lowest pitch: {'YES' if sad_lowest else 'NO'}")

        # Accuracy
        correct = sum([happy_highest, sad_lowest])
        print(f"\nAccuracy: {correct}/2 ({correct*50}%)")

if __name__ == "__main__":
    main()
