"""
Voice Clone Pipeline - Model Download Script
Downloads required models from Hugging Face.
"""

import os
import sys
from pathlib import Path

def download_models():
    print("=" * 50)
    print("Voice Clone Pipeline - Model Download")
    print("=" * 50)
    print()
    
    # Check for huggingface_hub
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download, hf_hub_download
    
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Download CSM-1B
    print("\n[1/3] Downloading CSM-1B (~4GB)...")
    try:
        snapshot_download(
            "sesame/csm-1b",
            local_dir=models_dir / "csm-1b",
            ignore_patterns=["*.md", "*.txt"],
        )
        print("  ✓ CSM-1B downloaded")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  Note: You may need to accept the model license at https://huggingface.co/sesame/csm-1b")
    
    # Download Whisper
    print("\n[2/3] Downloading Whisper large-v3 (~6GB)...")
    try:
        import whisper
        whisper.load_model("large-v3", download_root=str(models_dir / "whisper"))
        print("  ✓ Whisper downloaded")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  Note: Whisper will auto-download on first use")
    
    # Download Qwen2-Audio (optional - large)
    print("\n[3/3] Downloading Qwen2-Audio-7B (~14GB)...")
    print("  This is optional but recommended for prosody labeling.")
    
    response = input("  Download Qwen2-Audio? [y/N]: ").strip().lower()
    if response == 'y':
        try:
            snapshot_download(
                "Qwen/Qwen2-Audio-7B-Instruct",
                local_dir=models_dir / "qwen2-audio",
                ignore_patterns=["*.md", "*.txt"],
            )
            print("  ✓ Qwen2-Audio downloaded")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    else:
        print("  Skipped. Prosody labeling will use simpler analysis.")
    
    print("\n" + "=" * 50)
    print("Download complete!")
    print("=" * 50)
    print()
    print("Models are saved to:", models_dir)
    print()
    print("Next steps:")
    print("  1. Start backend: cd backend && python main.py")
    print("  2. Start frontend: cd frontend && npm run dev")
    print("  3. Open http://localhost:3000")


if __name__ == "__main__":
    download_models()
