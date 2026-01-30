#!/usr/bin/env python3
"""
Create a shortlist of most promising approaches for prosody/emotion TTS.
"""

# These are the most relevant to prosody/emotion control
SHORTLIST = {
    "PROSODY CONDITIONING (Your Core Goal)": [
        "train_prosody_conditioned.py",  # V7 baseline - already trained
        "train_prosody_hed.py",  # Hierarchical Emotion Distribution
        "train_sparse_keyframe.py",  # Sparse keyframe prosody
        "train_maskgct_prosody.py",  # MaskGCT prosody approach
        "prosody_flow.py",  # Flow-based prosody generation
    ],
    "EMOTION CONTROL (High Relevance)": [
        "train_emovoice.py",  # EmoVoice natural language control
        "spherical_emotion.py",  # Spherical emotion vectors
        "ece_tts_easv.py",  # EASV intensity control
        "emorl_tts.py",  # RL-based emotion optimization
        "emo_dpo.py",  # Preference optimization for emotion
    ],
    "STYLE TRANSFER (Your Current Approach)": [
        "train_lora_deepseek.py",  # Your current LoRA training
        "restyle_tts.py",  # ReStyle LoRA approach
        "train_style_lora.py",  # Style LoRA
    ],
    "ADVANCED CONDITIONING": [
        "activation_steering.py",  # Training-free emotion control
        "segment_aware_conditioning.py",  # Intra-utterance emotion
        "easv_intensity.py",  # Intensity control
    ],
}

# Archive these - not relevant to prosody/emotion goal
ARCHIVE = {
    "Codec/Compression (15)": "Not needed for your use case",
    "Disentanglement (8)": "Academic, not practical for your goal",
    "Multi-modal (3)": "Overkill for single-speaker cloning",
    "Flow/Diffusion codecs": "You're using CSM-1B, not building a codec",
}

def print_shortlist():
    print("=" * 80)
    print("RECOMMENDED SHORTLIST FOR EVALUATION")
    print("=" * 80)
    print("\nFocus on these 18 implementations (vs 129 total):")
    print()

    total = 0
    for category, scripts in SHORTLIST.items():
        print(f"\n{category}")
        print("-" * 80)
        for i, script in enumerate(scripts, 1):
            print(f"  {i}. {script}")
            total += 1

    print(f"\n{'=' * 80}")
    print(f"TOTAL: {total} implementations to evaluate")
    print(f"SAVINGS: Skip {129 - total} implementations ({100 * (129 - total) / 129:.0f}% reduction)")
    print(f"{'=' * 80}")

    print("\n\nARCHIVE THESE (Not Relevant):")
    print("-" * 80)
    for category, reason in ARCHIVE.items():
        print(f"  • {category}: {reason}")

    print(f"\n{'=' * 80}")
    print("NEXT STEPS")
    print(f"{'=' * 80}")
    print("""
OPTION A: Quick Manual Triage (FREE)
  1. Read the top 5 from each category
  2. Pick the 3 most promising based on paper/approach
  3. Run S4 on just those 3

OPTION B: Small-Scale S4 (CHEAPER)
  1. Run S4 on these 18 finalists only
  2. Rank by F0 separation + emotion accuracy
  3. Pick top 3 for deep dive

OPTION C: Archive & Start Fresh (PRAGMATIC)
  1. Archive all research code to docs/research_archive/
  2. Focus on improving your V7 baseline with 1-2 proven techniques
  3. Don't get distracted by 129 approaches

RECOMMENDATION: Option A or C
  - The research was exploratory, most won't beat V7
  - Pick 1-2 techniques that make theoretical sense
  - Train those on 4090, compare to V7
  - Don't spend $$ evaluating academic curiosities
""")

if __name__ == '__main__':
    print_shortlist()
