#!/usr/bin/env python3
"""
Evaluate temporal vs global keyframe prosody generation.

This script generates audio using keyframes in two modes:
1) Temporal (per-segment) prosody
2) Global (averaged) prosody

It then runs the prosody analyzer (without Qwen) on each output and
reports simple segment-level contour stats.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT / "inference"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from generate_with_prosody import ControllableVoiceGenerator
from prosody_analyzer import CompleteProsodyAnalyzer


def _load_keyframes(keyframes_arg: str) -> List[Dict]:
    path = Path(keyframes_arg)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(keyframes_arg)


def _segment_stats(contour: List[float], num_segments: int) -> List[float]:
    if not contour:
        return [0.0] * num_segments

    total = len(contour)
    seg_size = max(1, total // num_segments)
    stats = []
    for i in range(num_segments):
        start = i * seg_size
        end = total if i == num_segments - 1 else start + seg_size
        segment = contour[start:end]
        if segment:
            stats.append(float(sum(segment) / len(segment)))
        else:
            stats.append(0.0)
    return stats


def _analyze_audio(analyzer: CompleteProsodyAnalyzer, audio_path: Path, segments: int) -> Dict:
    result = analyzer.analyze(str(audio_path))
    prosody = result.to_dict()
    contour = prosody.get("contour", {})
    values = contour.get("smoothed") or contour.get("values") or []

    return {
        "acoustic": prosody.get("acoustic", {}),
        "rhythm": prosody.get("rhythm", {}),
        "contour_segment_means": _segment_stats(values, segments),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate temporal vs global keyframe prosody")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--keyframes", required=True, help="Keyframes JSON string or path to .json")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds for keyframes")
    parser.add_argument("--segments", type=int, default=4, help="Temporal segments")
    parser.add_argument("--output-dir", default="../inference/outputs", help="Output directory")

    parser.add_argument("--csm", default="../models/csm-1b", help="CSM model path")
    parser.add_argument("--prosody-ckpt", required=True, help="Prosody encoder checkpoint")
    parser.add_argument("--lora", help="LoRA adapter path")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda/mps)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keyframes = _load_keyframes(args.keyframes)

    generator = ControllableVoiceGenerator(
        csm_path=args.csm,
        prosody_checkpoint=args.prosody_ckpt,
        lora_adapter=args.lora,
        device=args.device,
    )

    # Temporal generation
    temporal_prosody = generator.get_keyframe_prosody(
        json.dumps(keyframes),
        text=args.text,
        duration_seconds=args.duration,
        use_temporal=True,
        num_segments=args.segments,
    )
    temporal_audio = generator.generate(args.text, temporal_prosody)
    temporal_path = output_dir / "temporal_keyframe.wav"
    generator.save_audio(temporal_audio, str(temporal_path))

    # Global generation
    global_prosody = generator.get_keyframe_prosody(
        json.dumps(keyframes),
        text=args.text,
        duration_seconds=args.duration,
        use_temporal=False,
        num_segments=args.segments,
    )
    global_audio = generator.generate(args.text, global_prosody)
    global_path = output_dir / "global_keyframe.wav"
    generator.save_audio(global_audio, str(global_path))

    # Analyze outputs (no Qwen)
    analyzer = CompleteProsodyAnalyzer(use_qwen=False)
    temporal_stats = _analyze_audio(analyzer, temporal_path, args.segments)
    global_stats = _analyze_audio(analyzer, global_path, args.segments)

    report = {
        "text": args.text,
        "keyframes": keyframes,
        "segments": args.segments,
        "temporal_audio": str(temporal_path),
        "global_audio": str(global_path),
        "temporal_stats": temporal_stats,
        "global_stats": global_stats,
    }

    report_path = output_dir / "keyframe_eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("Evaluation complete")
    print(f"Temporal audio: {temporal_path}")
    print(f"Global audio:   {global_path}")
    print(f"Report:         {report_path}")


if __name__ == "__main__":
    main()
