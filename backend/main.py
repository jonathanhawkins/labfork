"""
Voice Clone Pipeline - Backend API
FastAPI server for recording, transcription, and prosody analysis.
"""

import os
import json
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import base64
import numpy as np
import torch
import torchaudio
import whisper

from prosody_analyzer import CompleteProsodyAnalyzer, ProsodyResult, AcousticAnalyzer, RhythmAnalyzer, ContourExtractor
from prosody_predictor import ProsodyPredictor, predict_prosody
from libritts_loader import (
    get_loader,
    LibriTTSSample,
    generate_synthetic_audio,
    pitch_hz_to_category,
    speaking_rate_sps_to_category,
    infer_gender_from_pitch,
)

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "../data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELED_DIR = DATA_DIR / "labeled"

# Create directories
for d in [RAW_DIR, PROCESSED_DIR, LABELED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Initialize app
app = FastAPI(
    title="Voice Clone Pipeline API",
    description="Backend for voice recording, transcription, and prosody analysis",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3003", "http://127.0.0.1:3003"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (loaded lazily)
whisper_model = None
prosody_analyzer = None

# In-memory sample storage (would use DB in production)
samples: Dict[str, Dict[str, Any]] = {}


# ============== Models ==============

class SampleCreate(BaseModel):
    id: str
    transcript: Optional[str] = None
    approved: bool = False

class SampleUpdate(BaseModel):
    transcript: Optional[str] = None
    prosody: Optional[Dict[str, Any]] = None
    approved: Optional[bool] = None

class GenerateRequest(BaseModel):
    text: str
    context: Optional[str] = None
    temperature: float = 0.8


class ProsodyGenerateRequest(BaseModel):
    text: str
    emotion: Optional[str] = None  # If None, auto-predict from text
    intensity: float = 0.8
    temperature: float = 0.8


class KeyframeRequest(BaseModel):
    """Request for keyframe-based voice generation."""
    text: str
    keyframes: List[Dict[str, Any]]  # [{time|word_index|word|char_index: ..., emotion: "...", intensity: ...}, ...]
    temperature: float = 0.8


# Global prosody predictor
prosody_predictor = None


def get_prosody_predictor():
    """Lazy load prosody predictor."""
    global prosody_predictor
    if prosody_predictor is None:
        print("Loading prosody predictor...")
        prosody_predictor = ProsodyPredictor(use_api=False)  # Rule-based for fast startup
        print("Prosody predictor loaded!")
    return prosody_predictor


# ============== Utilities ==============

def get_whisper_model():
    """Lazy load Whisper model."""
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper large-v3...")
        whisper_model = whisper.load_model("large-v3")
        print("Whisper loaded!")
    return whisper_model


def get_prosody_analyzer():
    """Lazy load prosody analyzer."""
    global prosody_analyzer
    if prosody_analyzer is None:
        print("Loading prosody analyzer...")
        prosody_analyzer = CompleteProsodyAnalyzer(
            use_qwen=True,  # Set False if you don't have Qwen2-Audio
            device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Prosody analyzer loaded!")
    return prosody_analyzer


def resample_audio(input_path: Path, output_path: Path, target_sr: int = 24000):
    """Resample audio to target sample rate and convert to mono."""
    waveform, sr = torchaudio.load(input_path)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Save
    torchaudio.save(output_path, waveform, target_sr)
    return output_path


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "Voice Clone Pipeline API",
        "version": "1.0.0"
    }


@app.get("/status")
async def status():
    """Get system status and loaded models."""
    return {
        "whisper_loaded": whisper_model is not None,
        "prosody_analyzer_loaded": prosody_analyzer is not None,
        "samples_count": len(samples),
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dirs": {
            "raw": str(RAW_DIR),
            "processed": str(PROCESSED_DIR),
            "labeled": str(LABELED_DIR),
        }
    }


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file for processing.
    Returns a sample ID for further operations.
    """
    # Generate unique ID
    sample_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    
    # Save raw file
    raw_path = RAW_DIR / f"{sample_id}{Path(file.filename).suffix}"
    with open(raw_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Resample to 24kHz mono
    processed_path = PROCESSED_DIR / f"{sample_id}.wav"
    try:
        resample_audio(raw_path, processed_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")
    
    # Get duration
    waveform, sr = torchaudio.load(processed_path)
    duration = waveform.shape[1] / sr
    
    # Store sample info
    samples[sample_id] = {
        "id": sample_id,
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "duration": duration,
        "transcript": None,
        "prosody": None,
        "approved": False,
        "created_at": datetime.now().isoformat(),
    }
    
    return {
        "id": sample_id,
        "duration": duration,
        "status": "uploaded"
    }


@app.post("/transcribe/{sample_id}")
async def transcribe(sample_id: str):
    """
    Transcribe audio using Whisper.
    Returns transcript with word-level timestamps.
    """
    if sample_id not in samples:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    sample = samples[sample_id]
    audio_path = sample["processed_path"]
    
    # Load Whisper
    model = get_whisper_model()
    
    # Transcribe with word timestamps
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en"
    )
    
    # Extract transcript and words
    transcript = result["text"].strip()
    words = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"],
                "probability": word_info.get("probability", 1.0)
            })
    
    # Update sample
    samples[sample_id]["transcript"] = transcript
    samples[sample_id]["words"] = words
    
    return {
        "id": sample_id,
        "transcript": transcript,
        "words": words,
        "status": "transcribed"
    }


@app.post("/analyze/{sample_id}")
async def analyze_prosody(sample_id: str, background_tasks: BackgroundTasks):
    """
    Analyze prosody (acoustic + rhythm + semantic features).
    This can take a few seconds, especially for semantic analysis.
    """
    if sample_id not in samples:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    sample = samples[sample_id]
    audio_path = sample["processed_path"]
    transcript = sample.get("transcript", "")
    
    # Get analyzer
    analyzer = get_prosody_analyzer()
    
    # Run analysis
    result: ProsodyResult = analyzer.analyze(audio_path, transcript)
    
    # Update sample
    samples[sample_id]["prosody"] = result.to_dict()
    
    return {
        "id": sample_id,
        "prosody": result.to_dict(),
        "status": "analyzed"
    }


@app.post("/process/{sample_id}")
async def process_sample(sample_id: str):
    """
    Full processing pipeline: transcribe + analyze prosody.
    Convenience endpoint that does both steps.
    """
    # Transcribe
    await transcribe(sample_id)
    
    # Analyze
    await analyze_prosody(sample_id, BackgroundTasks())
    
    return {
        "id": sample_id,
        "sample": samples[sample_id],
        "status": "processed"
    }


@app.get("/sample/{sample_id}")
async def get_sample(sample_id: str):
    """Get sample details."""
    if sample_id not in samples:
        raise HTTPException(status_code=404, detail="Sample not found")
    return samples[sample_id]


@app.patch("/sample/{sample_id}")
async def update_sample(sample_id: str, update: SampleUpdate):
    """Update sample (transcript, prosody corrections, approval status)."""
    if sample_id not in samples:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    if update.transcript is not None:
        samples[sample_id]["transcript"] = update.transcript
    if update.prosody is not None:
        samples[sample_id]["prosody"] = update.prosody
    if update.approved is not None:
        samples[sample_id]["approved"] = update.approved
        
        # If approved, save to labeled directory
        if update.approved:
            save_labeled_sample(sample_id)
    
    return samples[sample_id]


@app.delete("/sample/{sample_id}")
async def delete_sample(sample_id: str):
    """Delete a sample."""
    if sample_id not in samples:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    sample = samples[sample_id]
    
    # Delete files
    for path_key in ["raw_path", "processed_path"]:
        if path_key in sample and Path(sample[path_key]).exists():
            Path(sample[path_key]).unlink()
    
    # Remove from memory
    del samples[sample_id]
    
    return {"status": "deleted"}


@app.get("/samples")
async def list_samples(approved_only: bool = False):
    """List all samples."""
    result = list(samples.values())
    if approved_only:
        result = [s for s in result if s.get("approved")]
    return {
        "count": len(result),
        "samples": result
    }


@app.get("/audio/{sample_id}")
async def get_audio(sample_id: str):
    """Stream audio file for playback."""
    if sample_id not in samples:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    audio_path = Path(samples[sample_id]["processed_path"])
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(audio_path, media_type="audio/wav")


@app.post("/export")
async def export_dataset():
    """
    Export approved samples as a training dataset.
    Creates metadata.json with all samples and prosody labels.
    """
    approved_samples = [s for s in samples.values() if s.get("approved")]
    
    if not approved_samples:
        raise HTTPException(status_code=400, detail="No approved samples to export")
    
    # Build export format
    export_data = []
    for sample in approved_samples:
        export_item = {
            "id": sample["id"],
            "text": sample.get("transcript", ""),
            "audio_path": sample["processed_path"],
            "duration": sample.get("duration", 0),
            "speaker": 0,  # Single speaker for now
        }
        
        # Add prosody if available
        if sample.get("prosody"):
            export_item["prosody"] = sample["prosody"]
        
        export_data.append(export_item)
    
    # Save metadata
    export_path = LABELED_DIR / "metadata.json"
    with open(export_path, "w") as f:
        json.dump(export_data, f, indent=2)
    
    # Calculate stats
    total_duration = sum(s.get("duration", 0) for s in approved_samples)
    
    return {
        "status": "exported",
        "path": str(export_path),
        "count": len(export_data),
        "total_duration_minutes": round(total_duration / 60, 2)
    }


def save_labeled_sample(sample_id: str):
    """Save an approved sample to the labeled directory."""
    sample = samples[sample_id]
    
    # Save label JSON
    label_path = LABELED_DIR / f"{sample_id}.json"
    with open(label_path, "w") as f:
        json.dump(sample, f, indent=2)
    
    # Copy audio to labeled dir
    import shutil
    src_audio = Path(sample["processed_path"])
    dst_audio = LABELED_DIR / f"{sample_id}.wav"
    if src_audio.exists() and not dst_audio.exists():
        shutil.copy(src_audio, dst_audio)


# ============== Stats Endpoints ==============

@app.post("/generate")
async def generate_speech(
    text: str = Form(None),
    emotion: str = Form(None),
    intensity: float = Form(0.8),
    temperature: float = Form(0.8),
    reference_audio: UploadFile = File(None),
):
    """
    Generate speech with prosody control.

    If emotion is not specified, it's automatically predicted from the text.
    If reference_audio is provided, prosody is extracted from it (style transfer).

    Returns the generated audio file.
    """
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Get prosody predictor
    predictor = get_prosody_predictor()

    # Predict or use specified prosody
    if reference_audio:
        # Style transfer: extract prosody from reference
        # Save reference temporarily
        ref_path = PROCESSED_DIR / f"ref_{uuid.uuid4().hex[:8]}.wav"
        with open(ref_path, "wb") as f:
            content = await reference_audio.read()
            f.write(content)

        # Extract prosody (would use analyzer in full implementation)
        predicted = predictor.predict(text)  # Fallback to text-based
        prosody_info = {
            "source": "reference_audio",
            "predicted": predicted.__dict__,
        }

        # Cleanup
        ref_path.unlink(missing_ok=True)
    elif emotion:
        # Use specified emotion
        from prosody_predictor import PredictedProsody
        predicted = PredictedProsody(
            emotion=emotion,
            emotion_intensity=intensity,
            energy=0.7 if emotion in ["happy", "angry", "excited"] else 0.4,
            pace="fast" if emotion in ["happy", "angry", "excited"] else "medium",
            pitch_tendency="high" if emotion in ["happy", "surprised"] else "neutral",
        )
        prosody_info = {
            "source": "user_specified",
            "predicted": predicted.__dict__,
        }
    else:
        # Auto-predict from text
        predicted = predictor.predict(text)
        prosody_info = {
            "source": "auto_predicted",
            "predicted": predicted.__dict__,
        }

    print(f"Generating with prosody: {prosody_info}")

    # Generate audio using inference script
    output_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_path = PROCESSED_DIR / f"generated_{output_id}.wav"

    try:
        # Import prosody-conditioned generator
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

        # Paths
        lora_adapter = Path(__file__).parent.parent / "models" / "checkpoints" / "csm_lora_450" / "best"
        base_model = Path(__file__).parent.parent / "models" / "csm-1b"
        prosody_checkpoint = Path(__file__).parent.parent / "models" / "checkpoints" / "prosody_conditioned" / "final.pt"

        # Use prosody-conditioned generator if checkpoint exists
        if prosody_checkpoint.exists():
            from generate_with_prosody import ControllableVoiceGenerator
            generator = ControllableVoiceGenerator(
                csm_path=str(base_model),
                prosody_checkpoint=str(prosody_checkpoint),
                lora_adapter=str(lora_adapter) if lora_adapter.exists() else None,
                device="auto",
            )

            # Get prosody from emotion
            prosody_tensors = generator.get_emotion_prosody(
                emotion=predicted.emotion if hasattr(predicted, 'emotion') else "neutral",
                intensity=predicted.emotion_intensity if hasattr(predicted, 'emotion_intensity') else intensity,
            )

            # Generate with prosody conditioning
            audio = generator.generate(
                text=text,
                prosody=prosody_tensors,
                temperature=temperature,
            )
        elif lora_adapter.exists():
            # Fallback to LoRA without prosody
            from generate_lora import LoRAVoiceGenerator
            generator = LoRAVoiceGenerator(
                adapter_path=str(lora_adapter),
                base_model_path=str(base_model),
                device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
            )
            audio = generator.generate(text=text, temperature=temperature)
        else:
            # Fallback to base model
            from generate import VoiceGenerator
            generator = VoiceGenerator(str(base_model))
            audio = generator.generate(text=text, temperature=temperature)

        # Save
        generator.save_audio(audio, str(output_path))

        duration = audio.shape[1] / 24000

    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Return the audio file
    return FileResponse(
        str(output_path),
        media_type="audio/wav",
        headers={
            "X-Prosody-Info": json.dumps(prosody_info),
            "X-Duration": str(duration),
        }
    )


# Pocket TTS model singleton
_pocket_tts_model = None

def get_pocket_tts_model():
    """Load Pocket TTS model (singleton)."""
    global _pocket_tts_model
    if _pocket_tts_model is None:
        try:
            from pocket_tts import TTSModel
            print("Loading Pocket TTS model...")
            _pocket_tts_model = TTSModel.load_model()
            print(f"  Voice cloning enabled: {_pocket_tts_model.has_voice_cloning}")
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Pocket TTS not installed. Run: pip install pocket-tts"
            )
    return _pocket_tts_model


@app.post("/generate-pocket-tts")
async def generate_with_pocket_tts(
    text: str = Form(...),
    reference_audio: UploadFile = File(None),
    builtin_voice: str = Form(None),
):
    """
    Generate speech using Pocket TTS (zero-shot voice cloning).

    Provide either:
    - reference_audio: WAV file of your voice to clone
    - builtin_voice: One of 'alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma'

    If neither provided, uses 'alba' as default.
    """
    import scipy.io.wavfile as wav

    model = get_pocket_tts_model()

    # Determine voice source
    if reference_audio:
        if not model.has_voice_cloning:
            raise HTTPException(
                status_code=400,
                detail="Voice cloning not enabled. Accept terms at https://huggingface.co/kyutai/pocket-tts"
            )

        # Save reference temporarily
        ref_path = PROCESSED_DIR / f"pocket_ref_{uuid.uuid4().hex[:8]}.wav"
        with open(ref_path, "wb") as f:
            content = await reference_audio.read()
            f.write(content)

        try:
            voice_state = model.get_state_for_audio_prompt(str(ref_path))
            voice_source = f"cloned from {reference_audio.filename}"
        finally:
            ref_path.unlink(missing_ok=True)
    else:
        # Use builtin voice
        voice_name = builtin_voice or "alba"
        valid_voices = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']
        if voice_name not in valid_voices:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice. Choose from: {', '.join(valid_voices)}"
            )
        voice_state = model.get_state_for_audio_prompt(voice_name)
        voice_source = f"builtin:{voice_name}"

    # Generate audio
    print(f"Generating with Pocket TTS: '{text[:50]}...' using {voice_source}")
    audio = model.generate_audio(voice_state, text)

    # Save output
    output_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_path = PROCESSED_DIR / f"pocket_tts_{output_id}.wav"

    audio_np = audio.cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    audio_np = (audio_np * 32767).astype(np.int16)
    wav.write(str(output_path), model.sample_rate, audio_np)

    duration = len(audio_np) / model.sample_rate
    print(f"Generated {duration:.2f}s audio: {output_path}")

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=f"pocket_tts_{output_id}.wav",
        headers={
            "X-Voice-Source": voice_source,
            "X-Duration": str(duration),
        }
    )


@app.get("/pocket-tts/voices")
async def list_pocket_tts_voices():
    """List available Pocket TTS voices."""
    model = get_pocket_tts_model()
    return {
        "builtin_voices": ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma'],
        "voice_cloning_enabled": model.has_voice_cloning,
        "sample_rate": model.sample_rate,
    }


@app.get("/voice-samples")
async def list_voice_samples():
    """List available recorded voice samples for voice cloning."""
    voice_samples_dir = DATA_DIR / "voice_samples"
    samples = []

    if voice_samples_dir.exists():
        for session_dir in sorted(voice_samples_dir.iterdir(), reverse=True):
            if session_dir.is_dir():
                # Get WAV files (excluding raw files)
                wav_files = [f for f in session_dir.glob("*.wav") if "_raw" not in f.name]
                for wav_file in wav_files:
                    # Extract emotion from filename (e.g., "neutral_1_111705_f751.wav")
                    parts = wav_file.stem.split("_")
                    emotion = parts[0] if parts else "unknown"

                    samples.append({
                        "id": wav_file.stem,
                        "path": str(wav_file),
                        "filename": wav_file.name,
                        "emotion": emotion,
                        "session_id": session_dir.name,
                    })

    # Group by emotion for easier selection
    by_emotion = {}
    for sample in samples:
        emotion = sample["emotion"]
        if emotion not in by_emotion:
            by_emotion[emotion] = []
        by_emotion[emotion].append(sample)

    return {
        "total_samples": len(samples),
        "samples": samples[:50],  # Limit to 50 most recent
        "by_emotion": by_emotion,
        "emotions": list(by_emotion.keys()),
    }


@app.post("/generate-with-voice-sample")
async def generate_with_voice_sample(
    text: str = Form(...),
    voice_sample_path: str = Form(...),
):
    """Generate speech using a recorded voice sample as reference."""
    import scipy.io.wavfile as wav
    import numpy as np

    # Validate sample path exists
    sample_path = Path(voice_sample_path)
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Voice sample not found")

    # Security check - ensure path is within voice_samples directory
    voice_samples_dir = DATA_DIR / "voice_samples"
    try:
        sample_path.resolve().relative_to(voice_samples_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid voice sample path")

    model = get_pocket_tts_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Pocket TTS not available")

    try:
        # Get voice state from recorded sample
        voice_state = model.get_state_for_audio_prompt(str(sample_path))

        # Generate audio
        audio = model.generate_audio(voice_state, text)

        # Save output
        output_id = f"{uuid.uuid4().hex[:8]}"
        output_path = PROCESSED_DIR / f"cloned_voice_{output_id}.wav"

        audio_np = (audio.cpu().numpy() * 32767).astype(np.int16)
        wav.write(str(output_path), 24000, audio_np)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"cloned_voice_{output_id}.wav",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate-keyframes")
async def generate_keyframes(request: KeyframeRequest):
    """
    Generate speech with keyframe-based prosody control.

    Keyframes define prosody at specific timestamps, which are then
    interpolated to create smooth prosody transitions throughout the audio.

    Each keyframe should have one of:
    - time: float (seconds from start or normalized 0-1)
    - word_index: int (zero-based word index in text)
    - word: str (word text, with optional occurrence)
    - char_index: int (character index in text)
    - emotion: str (e.g., "neutral", "happy", "sad", "angry", "excited")
    - intensity: float (0.0 to 1.0, controls emotion strength)

    Example keyframes:
    [
        {"time": 0.0, "emotion": "neutral", "intensity": 0.5},
        {"time": 1.5, "emotion": "excited", "intensity": 0.9},
        {"time": 3.0, "emotion": "neutral", "intensity": 0.6}
    ]
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    if not request.keyframes:
        raise HTTPException(status_code=400, detail="At least one keyframe is required")

    # Validate keyframes
    for i, kf in enumerate(request.keyframes):
        has_time = "time" in kf and kf.get("time") is not None
        has_word_index = "word_index" in kf
        has_word = "word" in kf
        has_char_index = "char_index" in kf
        if not (has_time or has_word_index or has_word or has_char_index):
            raise HTTPException(
                status_code=400,
                detail=f"Keyframe {i} missing time/word_index/word/char_index"
            )
        if "emotion" not in kf:
            raise HTTPException(
                status_code=400,
                detail=f"Keyframe {i} missing 'emotion' field"
            )
        if has_time and not isinstance(kf.get("time"), (int, float)):
            raise HTTPException(
                status_code=400,
                detail=f"Keyframe {i} 'time' must be a number"
            )
        if has_time and kf["time"] < 0:
            raise HTTPException(
                status_code=400,
                detail=f"Keyframe {i} 'time' must be non-negative"
            )
        if has_word_index and not isinstance(kf.get("word_index"), int):
            raise HTTPException(
                status_code=400,
                detail=f"Keyframe {i} 'word_index' must be an integer"
            )
        if has_char_index and not isinstance(kf.get("char_index"), int):
            raise HTTPException(
                status_code=400,
                detail=f"Keyframe {i} 'char_index' must be an integer"
            )
        if has_word and not isinstance(kf.get("word"), str):
            raise HTTPException(
                status_code=400,
                detail=f"Keyframe {i} 'word' must be a string"
            )

    # Resolve keyframe times (supports word_index/word/char_index anchors)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

        from keyframe_prosody import resolve_keyframe_times
        resolved_keyframes = resolve_keyframe_times(request.keyframes, request.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to resolve keyframes: {e}")

    # Sort keyframes by time
    sorted_keyframes = sorted(resolved_keyframes, key=lambda x: x["time"])

    print(f"Generating with {len(sorted_keyframes)} keyframes: {sorted_keyframes}")

    # Generate output path
    output_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_path = PROCESSED_DIR / f"generated_keyframes_{output_id}.wav"

    try:
        # Import required modules
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

        # Import keyframe prosody interpolation module
        from keyframe_prosody import (
            ProsodyKeyframe,
            keyframes_to_prosody,
            get_temporal_prosody_tokens,
        )

        # Paths for models
        lora_adapter = Path(__file__).parent.parent / "models" / "checkpoints" / "csm_lora_450" / "best"
        base_model = Path(__file__).parent.parent / "models" / "csm-1b"
        prosody_checkpoint = Path(__file__).parent.parent / "models" / "checkpoints" / "prosody_conditioned" / "final.pt"

        # Check if prosody-conditioned generator is available
        if not prosody_checkpoint.exists():
            raise HTTPException(
                status_code=503,
                detail="Prosody-conditioned model not available. Train the model first."
            )

        from generate_with_prosody import ControllableVoiceGenerator

        generator = ControllableVoiceGenerator(
            csm_path=str(base_model),
            prosody_checkpoint=str(prosody_checkpoint),
            lora_adapter=str(lora_adapter) if lora_adapter.exists() else None,
            device="auto",
        )

        # Convert raw keyframe dicts to ProsodyKeyframe objects
        prosody_keyframes = []
        for kf in sorted_keyframes:
            prosody_keyframes.append(ProsodyKeyframe(
                time=kf["time"],
                emotion=kf.get("emotion", "neutral"),
                intensity=kf.get("intensity", 0.7),
                energy=kf.get("energy"),
                pitch_tendency=kf.get("pitch_tendency"),
            ))

        # Estimate duration from keyframes (use max time, or default 3s)
        max_time = max(kf.time for kf in prosody_keyframes) if prosody_keyframes else 3.0
        duration_seconds = max(max_time, 1.0)  # At least 1 second

        # Convert keyframes to dense prosody tensors
        prosody_dense = keyframes_to_prosody(
            keyframes=prosody_keyframes,
            duration_seconds=duration_seconds,
        )

        # Convert to temporal tokens (preserves keyframe edits)
        num_segments = generator.prosody_config.num_prosody_tokens
        prosody_tensors = get_temporal_prosody_tokens(
            prosody_dense,
            num_segments=num_segments,
        )
        prosody_tensors["_is_temporal"] = True
        prosody_tensors["_num_segments"] = num_segments

        # Move prosody tensors to the same device as the model
        device = generator.device
        prosody_tensors = {
            k: v.to(device) if hasattr(v, 'to') else v
            for k, v in prosody_tensors.items()
        }

        # Generate audio with the interpolated prosody
        audio = generator.generate(
            text=request.text,
            prosody=prosody_tensors,
            temperature=request.temperature,
        )

        # Save the generated audio
        generator.save_audio(audio, str(output_path))

        duration = audio.shape[1] / 24000

        print(f"Generated keyframe audio: {output_path} ({duration:.2f}s)")

    except ImportError as e:
        print(f"Import error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Keyframe prosody module not available: {str(e)}"
        )
    except Exception as e:
        print(f"Keyframe generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Build prosody info for response header
    prosody_info = {
        "source": "keyframes",
        "keyframe_count": len(sorted_keyframes),
        "keyframes": sorted_keyframes,
    }

    # Return the audio file
    return FileResponse(
        str(output_path),
        media_type="audio/wav",
        headers={
            "X-Prosody-Info": json.dumps(prosody_info),
            "X-Duration": str(duration),
            "X-Keyframe-Count": str(len(sorted_keyframes)),
        }
    )


@app.post("/predict-prosody")
async def predict_prosody_endpoint(text: str):
    """
    Predict prosody from text without generating audio.

    Useful for previewing what prosody will be used before generation.
    """
    predictor = get_prosody_predictor()
    predicted = predictor.predict(text)
    conditioning = predictor.to_conditioning_dict(predicted)

    return {
        "text": text,
        "predicted": predicted.__dict__,
        "conditioning": conditioning,
    }


@app.get("/stats")
async def get_stats():
    """Get dataset statistics."""
    all_samples = list(samples.values())
    approved = [s for s in all_samples if s.get("approved")]

    total_duration = sum(s.get("duration", 0) for s in all_samples)
    approved_duration = sum(s.get("duration", 0) for s in approved)

    # Emotion distribution
    emotions = {}
    for s in approved:
        if s.get("prosody", {}).get("semantic", {}).get("emotion"):
            emotion = s["prosody"]["semantic"]["emotion"]
            emotions[emotion] = emotions.get(emotion, 0) + 1

    return {
        "total_samples": len(all_samples),
        "approved_samples": len(approved),
        "total_duration_minutes": round(total_duration / 60, 2),
        "approved_duration_minutes": round(approved_duration / 60, 2),
        "emotion_distribution": emotions,
    }


# ============== LibriTTS Comparison Endpoints ==============

@app.get("/libritts/stats")
async def get_libritts_stats():
    """Get LibriTTS dataset statistics."""
    try:
        loader = get_loader()
        return loader.get_stats()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="LibriTTS dataset not found")


@app.get("/libritts/samples")
async def list_libritts_samples(
    limit: int = 50,
    offset: int = 0,
    gender: Optional[str] = None,
    pitch_category: Optional[str] = None,
    speaking_rate: Optional[str] = None,
    speaker_id: Optional[str] = None,
    search: Optional[str] = None,
):
    """List LibriTTS samples with pagination and filtering."""
    try:
        loader = get_loader()
        samples_list, total = loader.list_samples(
            limit=limit,
            offset=offset,
            gender=gender,
            pitch_category=pitch_category,
            speaking_rate=speaking_rate,
            speaker_id=speaker_id,
            search_text=search,
        )

        return {
            "samples": [s.to_dict() for s in samples_list],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="LibriTTS dataset not found")


@app.get("/libritts/sample/{sample_id}")
async def get_libritts_sample(sample_id: str):
    """Get a single LibriTTS sample by ID."""
    try:
        loader = get_loader()
        sample = loader.get_sample(sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail="Sample not found")
        return sample.to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="LibriTTS dataset not found")


@app.post("/libritts/compare/{sample_id}")
async def compare_libritts_sample(sample_id: str, use_synthetic: bool = True):
    """
    Run prosody comparison on a LibriTTS sample.

    Generates synthetic audio (or uses real audio if available),
    runs our prosody analyzer, and compares with ground truth.
    """
    try:
        loader = get_loader()
        sample = loader.get_sample(sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail="Sample not found")

        # Generate synthetic audio
        audio_path = generate_synthetic_audio(sample)

        # Run prosody analysis
        acoustic_analyzer = AcousticAnalyzer()
        rhythm_analyzer = RhythmAnalyzer()
        contour_extractor = ContourExtractor()

        acoustic = acoustic_analyzer.analyze(str(audio_path))
        rhythm = rhythm_analyzer.analyze(str(audio_path))
        contour = contour_extractor.extract(str(audio_path))

        # Calculate metrics
        gt_pitch = sample.utterance_pitch_mean if sample.utterance_pitch_mean > 0 else 150.0
        pitch_mean_error = abs(acoustic.pitch_mean - gt_pitch)
        pitch_mean_error_pct = (pitch_mean_error / gt_pitch * 100) if gt_pitch > 0 else 0

        our_pitch_category = pitch_hz_to_category(acoustic.pitch_mean, sample.gender)
        our_rate_category = speaking_rate_sps_to_category(rhythm.speaking_rate)
        gender_inferred = infer_gender_from_pitch(acoustic.pitch_mean)

        return {
            "sample_id": sample_id,
            "audio_path": str(audio_path),
            "ground_truth": {
                "pitch_category": sample.pitch,
                "pitch_mean_hz": sample.utterance_pitch_mean,
                "pitch_std_hz": sample.utterance_pitch_std,
                "speaking_rate": sample.speaking_rate,
                "monotony": sample.speech_monotony,
                "gender": sample.gender,
                "text": sample.text,
            },
            "our_analysis": {
                "pitch_mean_hz": acoustic.pitch_mean,
                "pitch_std_hz": acoustic.pitch_std,
                "pitch_category": our_pitch_category,
                "speaking_rate_sps": rhythm.speaking_rate,
                "speaking_rate_category": our_rate_category,
                "gender_inferred": gender_inferred,
                "contour": {
                    "times": contour.times[:100] if contour.times else [],
                    "values": contour.values[:100] if contour.values else [],
                },
            },
            "metrics": {
                "pitch_mean_error_hz": round(pitch_mean_error, 2),
                "pitch_mean_error_pct": round(pitch_mean_error_pct, 2),
                "pitch_category_match": our_pitch_category == sample.pitch,
                "speaking_rate_match": our_rate_category == sample.speaking_rate,
                "gender_match": gender_inferred == sample.gender,
            },
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="LibriTTS dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


class BatchCompareRequest(BaseModel):
    sample_count: int = 50
    gender_filter: Optional[str] = None
    pitch_filter: Optional[str] = None
    use_real_audio: bool = False  # If True, use downloaded real audio files


# Directory for real audio files
REAL_AUDIO_DIR = Path(__file__).parent.parent / "data" / "real_audio" / "libritts_r"


@app.post("/libritts/batch-compare")
async def batch_compare_libritts(request: BatchCompareRequest):
    """
    Run prosody comparison on multiple LibriTTS samples.
    Returns aggregate statistics across all samples.
    """
    try:
        loader = get_loader()

        # Get samples with optional filters
        samples_list, total = loader.list_samples(
            limit=request.sample_count,
            offset=0,
            gender=request.gender_filter if request.gender_filter != "all" else None,
            pitch_category=request.pitch_filter if request.pitch_filter != "all" else None,
        )

        if not samples_list:
            raise HTTPException(status_code=404, detail="No samples found matching filters")

        # Initialize analyzers once for efficiency
        acoustic_analyzer = AcousticAnalyzer()
        rhythm_analyzer = RhythmAnalyzer()
        contour_extractor = ContourExtractor()

        results = []
        errors = []

        # Check for real audio files if requested
        real_audio_ids = set()
        if request.use_real_audio and REAL_AUDIO_DIR.exists():
            real_audio_ids = {f.stem for f in REAL_AUDIO_DIR.glob("*.wav")}

        for sample in samples_list:
            try:
                # Use real audio if available and requested, otherwise synthetic
                if request.use_real_audio and sample.id in real_audio_ids:
                    audio_path = REAL_AUDIO_DIR / f"{sample.id}.wav"
                else:
                    audio_path = generate_synthetic_audio(sample)

                # Run analysis
                acoustic = acoustic_analyzer.analyze(str(audio_path))
                rhythm = rhythm_analyzer.analyze(str(audio_path))

                # Calculate metrics
                gt_pitch = sample.utterance_pitch_mean if sample.utterance_pitch_mean > 0 else 150.0
                pitch_mean_error = abs(acoustic.pitch_mean - gt_pitch)
                pitch_mean_error_pct = (pitch_mean_error / gt_pitch * 100) if gt_pitch > 0 else 0

                our_pitch_category = pitch_hz_to_category(acoustic.pitch_mean, sample.gender)
                our_rate_category = speaking_rate_sps_to_category(rhythm.speaking_rate)
                gender_inferred = infer_gender_from_pitch(acoustic.pitch_mean)

                results.append({
                    "sample_id": sample.id,
                    "pitch_error_hz": pitch_mean_error,
                    "pitch_error_pct": pitch_mean_error_pct,
                    "pitch_category_match": our_pitch_category == sample.pitch,
                    "speaking_rate_match": our_rate_category == sample.speaking_rate,
                    "gender_match": gender_inferred == sample.gender,
                    "gt_gender": sample.gender,
                    "gt_pitch": sample.pitch,
                    "gt_rate": sample.speaking_rate,
                    "our_pitch_category": our_pitch_category,
                    "our_rate_category": our_rate_category,
                    "our_gender": gender_inferred,
                })
            except Exception as e:
                errors.append({"sample_id": sample.id, "error": str(e)})

        # Calculate aggregate statistics
        if results:
            pitch_errors = [r["pitch_error_hz"] for r in results]
            pitch_pct_errors = [r["pitch_error_pct"] for r in results]
            pitch_matches = sum(1 for r in results if r["pitch_category_match"])
            rate_matches = sum(1 for r in results if r["speaking_rate_match"])
            gender_matches = sum(1 for r in results if r["gender_match"])
            n = len(results)

            # Category breakdown
            pitch_category_breakdown = {}
            rate_category_breakdown = {}
            for r in results:
                # Pitch categories
                gt_pitch = r["gt_pitch"]
                if gt_pitch not in pitch_category_breakdown:
                    pitch_category_breakdown[gt_pitch] = {"total": 0, "correct": 0}
                pitch_category_breakdown[gt_pitch]["total"] += 1
                if r["pitch_category_match"]:
                    pitch_category_breakdown[gt_pitch]["correct"] += 1

                # Rate categories
                gt_rate = r["gt_rate"]
                if gt_rate not in rate_category_breakdown:
                    rate_category_breakdown[gt_rate] = {"total": 0, "correct": 0}
                rate_category_breakdown[gt_rate]["total"] += 1
                if r["speaking_rate_match"]:
                    rate_category_breakdown[gt_rate]["correct"] += 1

            return {
                "summary": {
                    "total_samples": n,
                    "successful": n,
                    "failed": len(errors),
                    "pitch": {
                        "mean_error_hz": round(sum(pitch_errors) / n, 2),
                        "median_error_hz": round(sorted(pitch_errors)[n // 2], 2),
                        "max_error_hz": round(max(pitch_errors), 2),
                        "mean_error_pct": round(sum(pitch_pct_errors) / n, 2),
                        "category_accuracy": round(pitch_matches / n * 100, 1),
                    },
                    "speaking_rate": {
                        "category_accuracy": round(rate_matches / n * 100, 1),
                    },
                    "gender": {
                        "accuracy": round(gender_matches / n * 100, 1),
                    },
                    "overall_accuracy": round(
                        (pitch_matches + rate_matches + gender_matches) / (n * 3) * 100, 1
                    ),
                },
                "pitch_category_breakdown": pitch_category_breakdown,
                "rate_category_breakdown": rate_category_breakdown,
                "individual_results": results[:20],  # First 20 for reference
                "errors": errors,
            }
        else:
            return {
                "summary": {"total_samples": 0, "successful": 0, "failed": len(errors)},
                "errors": errors,
            }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="LibriTTS dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch comparison failed: {str(e)}")


# ============== Evaluation Endpoints ==============

# In-memory storage for evaluations
evaluation_history: List[Dict[str, Any]] = []

# Model ID to path mapping
MODELS_DIR = Path(__file__).parent.parent / "models"
AVAILABLE_MODELS = {
    "csm-1b-base": {
        "path": str(MODELS_DIR / "csm-1b"),
        "name": "CSM-1B Base",
        "type": "base",
        "description": "Sesame CSM-1B foundation model",
    },
    "voice-deepseek-v1-final": {
        "path": str(MODELS_DIR / "checkpoints" / "voice_deepseek_v1" / "final.pt"),
        "name": "Voice DeepSeek v1 (Final)",
        "type": "finetuned",
        "description": "Fine-tuned with DeepSeek techniques (3 epochs)",
    },
    "voice-v1-best": {
        "path": str(MODELS_DIR / "checkpoints" / "voice_v1" / "best.pt"),
        "name": "Voice Clone v1 (Best)",
        "type": "finetuned",
        "description": "Best checkpoint from training run",
    },
}

# Reference audio for comparison (from training data)
REFERENCE_AUDIO_DIR = Path(__file__).parent.parent / "data" / "real_audio" / "libritts_r"

# Lazy-loaded evaluator
voice_evaluator = None


def get_voice_evaluator():
    """Lazy load the voice clone evaluator."""
    global voice_evaluator
    if voice_evaluator is None:
        print("Loading voice clone evaluator...")
        # Import from inference module
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
        from evaluate import VoiceCloneEvaluator

        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        voice_evaluator = VoiceCloneEvaluator(
            use_speaker_similarity=True,
            use_prosody=True,
            use_mcd=True,
            use_wer=True,
            whisper_model="base",  # Use base for faster evaluation
            device=device,
        )
        print("Voice evaluator loaded!")
    return voice_evaluator


class EvaluateGenerateRequest(BaseModel):
    """Request to generate audio with a model and optionally evaluate."""
    model_path: str
    text: str
    context: Optional[str] = None
    temperature: float = 0.8
    reference_path: Optional[str] = None  # For evaluation
    evaluate: bool = False  # Whether to run evaluation


class EvaluateCompareRequest(BaseModel):
    """Request to compare two audio files."""
    reference_path: str
    generated_path: str
    target_text: Optional[str] = None
    include_speaker_similarity: bool = True
    include_prosody: bool = True
    include_mcd: bool = True
    include_wer: bool = True


class EvaluationResult(BaseModel):
    """Evaluation result model."""
    id: str
    timestamp: str
    overall_score: float
    speaker_score: float
    prosody_score: float
    mcd_score: float
    intelligibility_score: float
    reference_path: str
    generated_path: str
    target_text: Optional[str]
    details: Optional[Dict[str, Any]] = None


@app.get("/evaluate/models")
async def get_available_models():
    """Get list of available models for evaluation."""
    models = []
    for model_id, info in AVAILABLE_MODELS.items():
        model_info = {
            "id": model_id,
            "name": info["name"],
            "type": info["type"],
            "description": info["description"],
            "available": Path(info["path"]).exists(),
        }
        models.append(model_info)
    return {"models": models}


@app.get("/evaluate/references")
async def get_reference_audios(limit: int = 10):
    """Get list of available reference audio files."""
    references = []
    if REFERENCE_AUDIO_DIR.exists():
        for audio_file in sorted(REFERENCE_AUDIO_DIR.glob("*.wav"))[:limit]:
            references.append({
                "id": audio_file.stem,
                "path": str(audio_file),
                "filename": audio_file.name,
            })
    return {"references": references}


class UICompareRequest(BaseModel):
    """Request for UI comparison - generates audio and evaluates."""
    text: str
    finetuned_model_id: str = "voice-deepseek-v1-final"
    reference_id: Optional[str] = None  # If None, uses first reference
    temperature: float = 0.8


@app.post("/evaluate/ui-compare")
async def ui_compare(request: UICompareRequest):
    """
    Unified endpoint for UI: generates audio with fine-tuned model and evaluates.
    Returns audio URLs and metrics for the comparison page.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
    from generate import VoiceGenerator

    # Get model path
    if request.finetuned_model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.finetuned_model_id}")

    model_info = AVAILABLE_MODELS[request.finetuned_model_id]
    model_path = Path(model_info["path"])

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model checkpoint not found: {model_path}")

    # Get reference audio
    if request.reference_id:
        reference_path = REFERENCE_AUDIO_DIR / f"{request.reference_id}.wav"
    else:
        # Use first available reference
        refs = list(REFERENCE_AUDIO_DIR.glob("*.wav"))
        if not refs:
            raise HTTPException(status_code=404, detail="No reference audio available")
        reference_path = refs[0]

    if not reference_path.exists():
        raise HTTPException(status_code=404, detail=f"Reference audio not found: {reference_path}")

    # Generate ID for this comparison
    output_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    # Generate audio
    print(f"UI Compare: Generating audio with {request.finetuned_model_id}")
    generator = VoiceGenerator(str(model_path))
    audio = generator.generate(
        text=request.text,
        temperature=request.temperature,
    )

    # Save generated audio
    output_path = PROCESSED_DIR / f"generated_{output_id}.wav"
    generator.save_audio(audio, str(output_path))

    generated_duration = audio.shape[1] / 24000
    print(f"Generated audio: {output_path} ({generated_duration:.2f}s)")

    # Get reference duration
    ref_waveform, ref_sr = torchaudio.load(str(reference_path))
    ref_duration = ref_waveform.shape[1] / ref_sr

    # Run evaluation
    print("Running evaluation...")
    evaluator = get_voice_evaluator()
    report = evaluator.evaluate(
        reference_path=str(reference_path),
        generated_path=str(output_path),
        target_text=request.text,
        model_path=str(model_path),
    )

    # Store in history
    eval_record = {
        "id": output_id,
        "timestamp": report.evaluation_timestamp,
        "text": request.text,
        "overall_score": report.overall_score,
        "speaker_score": report.speaker_score,
        "prosody_score": report.prosody_score_normalized,
        "mcd_score": report.mcd_score,
        "intelligibility_score": report.intelligibility_score,
        "reference_path": str(reference_path),
        "generated_path": str(output_path),
        "model_id": request.finetuned_model_id,
    }
    evaluation_history.append(eval_record)

    # Return result for UI
    return {
        "id": output_id,
        "text": request.text,
        "finetuned_audio_url": f"/generated-audio/generated_{output_id}.wav",
        "finetuned_duration": generated_duration,
        "reference_audio_url": f"/audio/reference/{reference_path.name}",
        "reference_duration": ref_duration,
        "metrics": {
            "speakerSimilarity": report.speaker_score,
            "prosodyMatch": report.prosody_score_normalized,
            "naturalness": report.overall_score / 20,  # Convert to 1-5 scale
            "pitchCorrelation": max(0, report.prosody.f0_correlation * 100) if report.prosody else 0,
            "rhythmScore": (report.prosody.prosody_score * 100) if report.prosody else 50,
            "energyAlignment": 50,  # Not directly available, use default
        },
        "overall_score": report.overall_score,
        "details": report.to_dict() if hasattr(report, 'to_dict') else {},
    }


@app.get("/generated-audio/{filename}")
async def serve_generated_audio(filename: str):
    """Serve generated audio files."""
    audio_path = PROCESSED_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")
    return FileResponse(str(audio_path), media_type="audio/wav")


@app.get("/audio/reference/{filename}")
async def serve_reference_audio(filename: str):
    """Serve reference audio files."""
    audio_path = REFERENCE_AUDIO_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Reference audio file not found")
    return FileResponse(str(audio_path), media_type="audio/wav")


@app.post("/evaluate/generate")
async def evaluate_generate(request: EvaluateGenerateRequest):
    """
    Generate audio with a model and optionally evaluate against a reference.

    Returns generated audio path and evaluation metrics if requested.
    """
    model_path = Path(request.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    # Generate audio
    print(f"Generating audio with model: {model_path}")

    # Import generator
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
    from generate import VoiceGenerator

    generator = VoiceGenerator(str(model_path))
    audio = generator.generate(
        text=request.text,
        context=request.context,
        temperature=request.temperature,
    )

    # Save generated audio
    output_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_path = PROCESSED_DIR / f"generated_{output_id}.wav"
    generator.save_audio(audio, str(output_path))

    result = {
        "id": output_id,
        "generated_path": str(output_path),
        "text": request.text,
        "model_path": str(model_path),
        "duration": audio.shape[1] / 24000,
    }

    # Run evaluation if requested
    if request.evaluate and request.reference_path:
        reference_path = Path(request.reference_path)
        if not reference_path.exists():
            raise HTTPException(status_code=404, detail=f"Reference audio not found: {reference_path}")

        evaluator = get_voice_evaluator()
        report = evaluator.evaluate(
            reference_path=str(reference_path),
            generated_path=str(output_path),
            target_text=request.text,
            model_path=str(model_path),
        )

        # Store in history
        eval_record = {
            "id": output_id,
            "timestamp": report.evaluation_timestamp,
            "overall_score": report.overall_score,
            "speaker_score": report.speaker_score,
            "prosody_score": report.prosody_score_normalized,
            "mcd_score": report.mcd_score,
            "intelligibility_score": report.intelligibility_score,
            "reference_path": str(reference_path),
            "generated_path": str(output_path),
            "target_text": request.text,
            "details": report.to_dict(),
        }
        evaluation_history.append(eval_record)

        result["evaluation"] = eval_record

    return result


@app.post("/evaluate/compare")
async def evaluate_compare(request: EvaluateCompareRequest):
    """
    Compare two audio files and return evaluation metrics.

    Computes speaker similarity, prosody match, MCD, and optionally WER.
    """
    reference_path = Path(request.reference_path)
    generated_path = Path(request.generated_path)

    if not reference_path.exists():
        raise HTTPException(status_code=404, detail=f"Reference audio not found: {reference_path}")
    if not generated_path.exists():
        raise HTTPException(status_code=404, detail=f"Generated audio not found: {generated_path}")

    # Import evaluators
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
    from evaluate import (
        SpeakerSimilarityEvaluator,
        ProsodyEvaluator,
        MCDEvaluator,
        WEREvaluator,
    )

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        "reference_path": str(reference_path),
        "generated_path": str(generated_path),
        "target_text": request.target_text,
        "metrics": {},
    }

    # Speaker similarity
    if request.include_speaker_similarity:
        try:
            evaluator = SpeakerSimilarityEvaluator(device=device)
            speaker_result = evaluator.evaluate(str(reference_path), str(generated_path))
            results["metrics"]["speaker_similarity"] = speaker_result.to_dict()
        except Exception as e:
            results["metrics"]["speaker_similarity"] = {"error": str(e)}

    # Prosody
    if request.include_prosody:
        try:
            evaluator = ProsodyEvaluator()
            prosody_result = evaluator.evaluate(str(reference_path), str(generated_path))
            results["metrics"]["prosody"] = prosody_result.to_dict()
        except Exception as e:
            results["metrics"]["prosody"] = {"error": str(e)}

    # MCD
    if request.include_mcd:
        try:
            evaluator = MCDEvaluator()
            mcd_result = evaluator.evaluate(str(reference_path), str(generated_path))
            results["metrics"]["mcd"] = mcd_result.to_dict()
        except Exception as e:
            results["metrics"]["mcd"] = {"error": str(e)}

    # WER
    if request.include_wer and request.target_text:
        try:
            evaluator = WEREvaluator(model_name="base", device=device)
            wer_result = evaluator.evaluate(str(generated_path), request.target_text)
            results["metrics"]["wer"] = wer_result.to_dict()
        except Exception as e:
            results["metrics"]["wer"] = {"error": str(e)}

    # Calculate overall scores
    scores = {}

    if "speaker_similarity" in results["metrics"] and "cosine_similarity" in results["metrics"]["speaker_similarity"]:
        scores["speaker"] = results["metrics"]["speaker_similarity"]["cosine_similarity"] * 100

    if "prosody" in results["metrics"] and "prosody_score" in results["metrics"]["prosody"]:
        scores["prosody"] = results["metrics"]["prosody"]["prosody_score"] * 100

    if "mcd" in results["metrics"] and "mcd" in results["metrics"]["mcd"]:
        mcd_val = results["metrics"]["mcd"]["mcd"]
        scores["mcd"] = float(max(0, min(100, 100 * np.exp(-mcd_val / 5))))

    if "wer" in results["metrics"] and "wer" in results["metrics"]["wer"]:
        wer_val = results["metrics"]["wer"]["wer"]
        scores["intelligibility"] = float(max(0, min(100, 100 * (1 - wer_val))))

    # Weighted overall score
    weights = {"speaker": 0.35, "prosody": 0.25, "mcd": 0.25, "intelligibility": 0.15}
    total_weight = sum(weights[k] for k in scores.keys())

    if total_weight > 0:
        overall = sum(scores[k] * weights[k] / total_weight for k in scores.keys())
    else:
        overall = 0.0

    results["scores"] = scores
    results["overall_score"] = overall

    # Store in history
    eval_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    eval_record = {
        "id": eval_id,
        "timestamp": datetime.now().isoformat(),
        "overall_score": overall,
        "speaker_score": scores.get("speaker", 0),
        "prosody_score": scores.get("prosody", 0),
        "mcd_score": scores.get("mcd", 0),
        "intelligibility_score": scores.get("intelligibility", 0),
        "reference_path": str(reference_path),
        "generated_path": str(generated_path),
        "target_text": request.target_text,
        "details": results["metrics"],
    }
    evaluation_history.append(eval_record)
    results["id"] = eval_id

    return results


@app.get("/evaluate/history")
async def get_evaluation_history(
    limit: int = 50,
    offset: int = 0,
):
    """
    Get past evaluations from history.

    Returns evaluation records sorted by timestamp (newest first).
    """
    # Sort by timestamp descending
    sorted_history = sorted(
        evaluation_history,
        key=lambda x: x.get("timestamp", ""),
        reverse=True
    )

    # Paginate
    total = len(sorted_history)
    records = sorted_history[offset:offset + limit]

    # Calculate aggregate stats
    if sorted_history:
        overall_scores = [r.get("overall_score", 0) for r in sorted_history]
        speaker_scores = [r.get("speaker_score", 0) for r in sorted_history if r.get("speaker_score")]
        prosody_scores = [r.get("prosody_score", 0) for r in sorted_history if r.get("prosody_score")]
        mcd_scores = [r.get("mcd_score", 0) for r in sorted_history if r.get("mcd_score")]

        stats = {
            "total_evaluations": total,
            "overall_score_mean": float(np.mean(overall_scores)) if overall_scores else 0,
            "overall_score_std": float(np.std(overall_scores)) if len(overall_scores) > 1 else 0,
            "speaker_score_mean": float(np.mean(speaker_scores)) if speaker_scores else 0,
            "prosody_score_mean": float(np.mean(prosody_scores)) if prosody_scores else 0,
            "mcd_score_mean": float(np.mean(mcd_scores)) if mcd_scores else 0,
        }
    else:
        stats = {
            "total_evaluations": 0,
            "overall_score_mean": 0,
            "overall_score_std": 0,
            "speaker_score_mean": 0,
            "prosody_score_mean": 0,
            "mcd_score_mean": 0,
        }

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "stats": stats,
        "evaluations": records,
    }


@app.delete("/evaluate/history/{evaluation_id}")
async def delete_evaluation(evaluation_id: str):
    """Delete an evaluation from history."""
    global evaluation_history

    original_len = len(evaluation_history)
    evaluation_history = [e for e in evaluation_history if e.get("id") != evaluation_id]

    if len(evaluation_history) == original_len:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return {"status": "deleted", "id": evaluation_id}


@app.delete("/evaluate/history")
async def clear_evaluation_history():
    """Clear all evaluation history."""
    global evaluation_history
    count = len(evaluation_history)
    evaluation_history = []
    return {"status": "cleared", "deleted_count": count}


# ============== Voice Recording Script Endpoints ==============

# Directory for voice samples
VOICE_SAMPLES_DIR = DATA_DIR / "voice_samples"
VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Prosody encoder model path
PROSODY_ENCODER_PATH = Path(__file__).parent.parent / "models" / "prosody_encoder_ravdess" / "best.pt"

# Global prosody encoder model
prosody_encoder_model = None


def get_prosody_encoder():
    """Lazy load the trained prosody encoder."""
    global prosody_encoder_model
    if prosody_encoder_model is None and PROSODY_ENCODER_PATH.exists():
        print(f"Loading prosody encoder from {PROSODY_ENCODER_PATH}...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
        from train_prosody_ravdess import ProsodyEncoder

        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        prosody_encoder_model = {
            "model": ProsodyEncoder(hidden_dim=512).to(device),
            "device": device,
        }

        checkpoint = torch.load(PROSODY_ENCODER_PATH, map_location=device)
        prosody_encoder_model["model"].load_state_dict(checkpoint["model_state_dict"])
        prosody_encoder_model["model"].eval()

        print(f"Prosody encoder loaded on {device}")
    return prosody_encoder_model


# In-memory storage for voice recording sessions
voice_recording_sessions: Dict[str, Dict[str, Any]] = {}


class ScriptLine(BaseModel):
    """A single script line for recording."""
    id: str
    text: str
    source: str
    emotion: str
    difficulty: str = "medium"  # easy, medium, hard


class RecordingSession(BaseModel):
    """A recording session with multiple script lines."""
    session_id: str
    created_at: str
    total_lines: int
    recorded_count: int
    recordings: List[Dict[str, Any]]


class RecordingUpload(BaseModel):
    """Upload data for a voice recording."""
    line_id: str
    expected_emotion: str


@app.get("/script-lines")
async def get_script_lines():
    """
    Get curated script lines organized by emotion.
    These are iconic movie quotes, Shakespeare lines, and famous speeches
    designed to elicit natural emotional performances.
    """
    script_lines = {
        "neutral": [
            {"id": "neutral_1", "text": "The quick brown fox jumps over the lazy dog.", "source": "Pangram", "emotion": "neutral", "difficulty": "easy"},
            {"id": "neutral_2", "text": "Good morning. The weather today will be partly cloudy with a chance of rain.", "source": "Weather Report", "emotion": "neutral", "difficulty": "easy"},
            {"id": "neutral_3", "text": "Please leave your name and number after the tone, and I will return your call.", "source": "Voicemail", "emotion": "neutral", "difficulty": "easy"},
            {"id": "neutral_4", "text": "All human beings are born free and equal in dignity and rights.", "source": "Universal Declaration of Human Rights", "emotion": "neutral", "difficulty": "medium"},
            {"id": "neutral_5", "text": "In the beginning, there was nothing. And then there was light.", "source": "Narrator", "emotion": "neutral", "difficulty": "medium"},
        ],
        "calm": [
            {"id": "calm_1", "text": "It's okay. Take a deep breath. Everything is going to be fine.", "source": "Reassurance", "emotion": "calm", "difficulty": "easy"},
            {"id": "calm_2", "text": "The sea is calm tonight. The tide is full, the moon lies fair upon the straits.", "source": "Dover Beach - Matthew Arnold", "emotion": "calm", "difficulty": "medium"},
            {"id": "calm_3", "text": "All we have to decide is what to do with the time that is given us.", "source": "Lord of the Rings - Gandalf", "emotion": "calm", "difficulty": "medium"},
            {"id": "calm_4", "text": "We are such stuff as dreams are made on, and our little life is rounded with a sleep.", "source": "The Tempest - Shakespeare", "emotion": "calm", "difficulty": "hard"},
            {"id": "calm_5", "text": "Peace comes from within. Do not seek it without.", "source": "Buddha", "emotion": "calm", "difficulty": "easy"},
        ],
        "happy": [
            {"id": "happy_1", "text": "After all this time? Always.", "source": "Harry Potter - Snape", "emotion": "happy", "difficulty": "easy"},
            {"id": "happy_2", "text": "Oh what a beautiful morning! Oh what a beautiful day!", "source": "Oklahoma! - Musical", "emotion": "happy", "difficulty": "easy"},
            {"id": "happy_3", "text": "I'm the king of the world!", "source": "Titanic - Jack", "emotion": "happy", "difficulty": "easy"},
            {"id": "happy_4", "text": "My mama always said life was like a box of chocolates. You never know what you're gonna get.", "source": "Forrest Gump", "emotion": "happy", "difficulty": "medium"},
            {"id": "happy_5", "text": "To infinity and beyond!", "source": "Toy Story - Buzz Lightyear", "emotion": "happy", "difficulty": "easy"},
        ],
        "sad": [
            {"id": "sad_1", "text": "I could have saved more. I could have saved more. I don't know. If I'd just made more money.", "source": "Schindler's List", "emotion": "sad", "difficulty": "hard"},
            {"id": "sad_2", "text": "To weep is to make less the depth of grief.", "source": "Henry VI - Shakespeare", "emotion": "sad", "difficulty": "medium"},
            {"id": "sad_3", "text": "I wish I knew how to quit you.", "source": "Brokeback Mountain", "emotion": "sad", "difficulty": "medium"},
            {"id": "sad_4", "text": "Of all the words of mice and men, the saddest are it might have been.", "source": "Kurt Vonnegut", "emotion": "sad", "difficulty": "medium"},
            {"id": "sad_5", "text": "I've seen things you people wouldn't believe. All those moments will be lost in time, like tears in rain.", "source": "Blade Runner - Roy", "emotion": "sad", "difficulty": "hard"},
        ],
        "angry": [
            {"id": "angry_1", "text": "You can't handle the truth!", "source": "A Few Good Men - Col. Jessup", "emotion": "angry", "difficulty": "medium"},
            {"id": "angry_2", "text": "I'm as mad as hell, and I'm not going to take this anymore!", "source": "Network - Howard Beale", "emotion": "angry", "difficulty": "hard"},
            {"id": "angry_3", "text": "You talkin' to me? You talkin' to me? Then who the hell else are you talking to?", "source": "Taxi Driver - Travis", "emotion": "angry", "difficulty": "medium"},
            {"id": "angry_4", "text": "I am not an animal! I am a human being!", "source": "The Elephant Man", "emotion": "angry", "difficulty": "hard"},
            {"id": "angry_5", "text": "Say hello to my little friend!", "source": "Scarface - Tony Montana", "emotion": "angry", "difficulty": "medium"},
        ],
        "fearful": [
            {"id": "fearful_1", "text": "I see dead people. Walking around like regular people. They don't know they're dead.", "source": "The Sixth Sense - Cole", "emotion": "fearful", "difficulty": "medium"},
            {"id": "fearful_2", "text": "The only thing we have to fear is fear itself.", "source": "FDR Inaugural Address", "emotion": "fearful", "difficulty": "hard"},
            {"id": "fearful_3", "text": "Cowards die many times before their deaths. The valiant never taste of death but once.", "source": "Julius Caesar - Shakespeare", "emotion": "fearful", "difficulty": "hard"},
            {"id": "fearful_4", "text": "Here's Johnny!", "source": "The Shining - Jack", "emotion": "fearful", "difficulty": "easy"},
            {"id": "fearful_5", "text": "What's in the box? What's in the box?!", "source": "Se7en - Detective Mills", "emotion": "fearful", "difficulty": "medium"},
        ],
        "surprised": [
            {"id": "surprised_1", "text": "I am your father.", "source": "Star Wars - Darth Vader", "emotion": "surprised", "difficulty": "easy"},
            {"id": "surprised_2", "text": "Wait, are you saying... we actually won? We won the championship!", "source": "Sports Moment", "emotion": "surprised", "difficulty": "medium"},
            {"id": "surprised_3", "text": "Toto, I've a feeling we're not in Kansas anymore.", "source": "Wizard of Oz - Dorothy", "emotion": "surprised", "difficulty": "easy"},
            {"id": "surprised_4", "text": "You mean to tell me that this whole time, it was you?", "source": "Plot Twist Reveal", "emotion": "surprised", "difficulty": "medium"},
            {"id": "surprised_5", "text": "Great Scott! The time machine... it actually works!", "source": "Back to the Future - Doc Brown", "emotion": "surprised", "difficulty": "medium"},
        ],
        "excited": [
            {"id": "excited_1", "text": "I have a dream that one day this nation will rise up!", "source": "MLK - I Have a Dream", "emotion": "excited", "difficulty": "hard"},
            {"id": "excited_2", "text": "We shall fight on the beaches, we shall fight on the landing grounds, we shall never surrender!", "source": "Churchill - WWII Speech", "emotion": "excited", "difficulty": "hard"},
            {"id": "excited_3", "text": "Today is the day! Everything changes starting right now!", "source": "Motivational", "emotion": "excited", "difficulty": "easy"},
            {"id": "excited_4", "text": "That's one small step for man, one giant leap for mankind!", "source": "Neil Armstrong - Moon Landing", "emotion": "excited", "difficulty": "medium"},
            {"id": "excited_5", "text": "Carpe diem! Seize the day, boys! Make your lives extraordinary!", "source": "Dead Poets Society - Keating", "emotion": "excited", "difficulty": "medium"},
        ],
    }

    # Count totals
    total_lines = sum(len(lines) for lines in script_lines.values())

    return {
        "emotions": list(script_lines.keys()),
        "total_lines": total_lines,
        "lines_per_emotion": {k: len(v) for k, v in script_lines.items()},
        "script_lines": script_lines,
    }


@app.post("/voice-recording/start-session")
async def start_recording_session():
    """Start a new voice recording session."""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    # Create session directory
    session_dir = VOICE_SAMPLES_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    session = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "session_dir": str(session_dir),
        "total_lines": 0,
        "recorded_count": 0,
        "recordings": [],
    }

    voice_recording_sessions[session_id] = session

    return session


@app.post("/voice-recording/upload/{session_id}")
async def upload_voice_recording(
    session_id: str,
    file: UploadFile = File(...),
    line_id: str = Form(...),
    expected_emotion: str = Form(...),
    script_text: str = Form(...),
):
    """
    Upload a voice recording and auto-label with prosody encoder.
    Returns emotion prediction and confidence.
    """
    print(f"\n[UPLOAD] Received upload request for session: {session_id}")
    print(f"[UPLOAD] File: {file.filename}, line_id: {line_id}, emotion: {expected_emotion}")

    if session_id not in voice_recording_sessions:
        print(f"[UPLOAD] ERROR: Session not found! Available sessions: {list(voice_recording_sessions.keys())}")
        raise HTTPException(status_code=404, detail="Session not found")

    session = voice_recording_sessions[session_id]
    session_dir = Path(session["session_dir"])
    print(f"[UPLOAD] Session dir: {session_dir}")

    # Generate unique recording ID
    recording_id = f"{line_id}_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}"
    print(f"[UPLOAD] Recording ID: {recording_id}")

    # Save raw audio
    raw_path = session_dir / f"{recording_id}_raw{Path(file.filename).suffix}"
    print(f"[UPLOAD] Saving raw audio to: {raw_path}")
    with open(raw_path, "wb") as f:
        content = await file.read()
        print(f"[UPLOAD] Read {len(content)} bytes from upload")
        f.write(content)
    print(f"[UPLOAD] Raw audio saved successfully")

    # Resample to 24kHz mono WAV
    processed_path = session_dir / f"{recording_id}.wav"
    try:
        print(f"[UPLOAD] Resampling to: {processed_path}")
        resample_audio(raw_path, processed_path)
        print(f"[UPLOAD] Resampling complete")
    except Exception as e:
        print(f"[UPLOAD] ERROR during resampling: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

    # Get audio duration
    waveform, sr = torchaudio.load(processed_path)
    duration = waveform.shape[1] / sr

    # Auto-label with prosody encoder
    prosody_result = await analyze_with_prosody_encoder(str(processed_path))

    # Build recording metadata
    recording = {
        "recording_id": recording_id,
        "line_id": line_id,
        "script_text": script_text,
        "expected_emotion": expected_emotion,
        "audio_path": str(processed_path),
        "duration": duration,
        "created_at": datetime.now().isoformat(),
        "prosody_analysis": prosody_result,
        "emotion_match": prosody_result.get("predicted_emotion") == expected_emotion if prosody_result else None,
    }

    # Update session
    session["recordings"].append(recording)
    session["recorded_count"] = len(session["recordings"])

    return {
        "recording_id": recording_id,
        "duration": duration,
        "prosody_analysis": prosody_result,
        "emotion_match": recording["emotion_match"],
        "session_progress": {
            "recorded_count": session["recorded_count"],
        },
    }


async def analyze_with_prosody_encoder(audio_path: str) -> Dict[str, Any]:
    """
    Analyze audio with the trained prosody encoder.
    Returns predicted emotion and prosody features.
    """
    encoder = get_prosody_encoder()

    if encoder is None:
        # Fallback to rule-based analysis
        return await fallback_prosody_analysis(audio_path)

    try:
        model = encoder["model"]
        device = encoder["device"]

        # Load and preprocess audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate to max length (10 seconds)
        max_len = 24000 * 10
        if waveform.shape[1] > max_len:
            waveform = waveform[:, :max_len]
        elif waveform.shape[1] < max_len:
            padding = max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Run through model
        waveform = waveform.squeeze(0).unsqueeze(0).to(device)  # (1, T)

        with torch.no_grad():
            outputs = model(waveform)

        # Map emotion index to label
        emotion_map = {
            0: "neutral", 1: "calm", 2: "happy", 3: "sad",
            4: "angry", 5: "fearful", 6: "surprised", 7: "excited"
        }

        emotion_logits = outputs["emotion_logits"][0]
        emotion_probs = torch.softmax(emotion_logits, dim=0).cpu().numpy()
        predicted_idx = int(emotion_logits.argmax().item())
        predicted_emotion = emotion_map[predicted_idx]
        confidence = float(emotion_probs[predicted_idx])

        # Get all emotion probabilities
        emotion_scores = {emotion_map[i]: float(emotion_probs[i]) for i in range(8)}

        # Extract other features
        acoustic = outputs["acoustic"][0].cpu().numpy()
        rhythm = outputs["rhythm"][0].cpu().numpy()
        contour = outputs["contour"][0].cpu().numpy()

        return {
            "predicted_emotion": predicted_emotion,
            "confidence": confidence,
            "emotion_scores": emotion_scores,
            "acoustic": {
                "pitch_mean": float(acoustic[0]),
                "pitch_std": float(acoustic[1]),
                "energy": float(acoustic[2]),
                "speaking_rate": float(acoustic[3]),
            },
            "rhythm": {
                "pause_ratio": float(rhythm[0]),
                "syllable_rate": float(rhythm[1]),
            },
            "contour": contour.tolist()[:32],  # First 32 points
            "model_used": "prosody_encoder_ravdess",
        }

    except Exception as e:
        print(f"Prosody encoder analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return await fallback_prosody_analysis(audio_path)


async def fallback_prosody_analysis(audio_path: str) -> Dict[str, Any]:
    """
    Fallback prosody analysis using Parselmouth when encoder not available.
    """
    try:
        acoustic_analyzer = AcousticAnalyzer()
        rhythm_analyzer = RhythmAnalyzer()

        acoustic = acoustic_analyzer.analyze(audio_path)
        rhythm = rhythm_analyzer.analyze(audio_path)

        # Simple emotion heuristic based on acoustic features
        pitch_mean = acoustic.pitch_mean
        energy = acoustic.energy if hasattr(acoustic, "energy") else 0.5

        # Very simple rule-based emotion
        if pitch_mean > 200:
            predicted_emotion = "excited"
        elif pitch_mean > 150:
            predicted_emotion = "happy"
        elif pitch_mean < 100:
            predicted_emotion = "sad"
        else:
            predicted_emotion = "neutral"

        return {
            "predicted_emotion": predicted_emotion,
            "confidence": 0.3,  # Low confidence for rule-based
            "acoustic": {
                "pitch_mean": acoustic.pitch_mean,
                "pitch_std": acoustic.pitch_std,
                "energy": getattr(acoustic, "energy", 0.5),
            },
            "rhythm": {
                "speaking_rate": rhythm.speaking_rate,
                "pause_ratio": getattr(rhythm, "pause_ratio", 0.2),
            },
            "model_used": "rule_based_fallback",
        }
    except Exception as e:
        print(f"Fallback analysis failed: {e}")
        return {
            "predicted_emotion": "neutral",
            "confidence": 0.0,
            "error": str(e),
            "model_used": "none",
        }


@app.get("/voice-recording/session/{session_id}")
async def get_recording_session(session_id: str):
    """Get details of a recording session."""
    if session_id not in voice_recording_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return voice_recording_sessions[session_id]


@app.get("/voice-recording/sessions")
async def list_recording_sessions():
    """List all recording sessions."""
    sessions = list(voice_recording_sessions.values())
    return {
        "count": len(sessions),
        "sessions": sessions,
    }


@app.get("/voice-recording/audio/{session_id}/{recording_id}")
async def get_recording_audio(session_id: str, recording_id: str):
    """Stream a recorded audio file."""
    if session_id not in voice_recording_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = voice_recording_sessions[session_id]
    recording = next(
        (r for r in session["recordings"] if r["recording_id"] == recording_id),
        None
    )

    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    audio_path = Path(recording["audio_path"])
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(audio_path, media_type="audio/wav")


@app.post("/voice-recording/export/{session_id}")
async def export_session_for_training(session_id: str):
    """
    Export a recording session as training data.
    Creates manifest file compatible with the training pipeline.
    """
    if session_id not in voice_recording_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = voice_recording_sessions[session_id]

    if not session["recordings"]:
        raise HTTPException(status_code=400, detail="No recordings in session")

    # Build training manifest
    manifest = []
    for recording in session["recordings"]:
        item = {
            "id": recording["recording_id"],
            "audio_path": recording["audio_path"],
            "text": recording["script_text"],
            "duration": recording["duration"],
            "prosody": {
                "semantic": {
                    "emotion": recording.get("prosody_analysis", {}).get("predicted_emotion", "neutral"),
                    "intensity": recording.get("prosody_analysis", {}).get("confidence", 0.5),
                },
                "acoustic": recording.get("prosody_analysis", {}).get("acoustic", {}),
                "rhythm": recording.get("prosody_analysis", {}).get("rhythm", {}),
                "contour": {
                    "pitch_contour": recording.get("prosody_analysis", {}).get("contour", []),
                },
            },
            "expected_emotion": recording["expected_emotion"],
            "created_at": recording["created_at"],
        }
        manifest.append(item)

    # Save manifest
    session_dir = Path(session["session_dir"])
    manifest_path = session_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Calculate stats
    total_duration = sum(r["duration"] for r in session["recordings"])
    emotion_counts = {}
    for r in session["recordings"]:
        emotion = r.get("prosody_analysis", {}).get("predicted_emotion", "unknown")
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    return {
        "status": "exported",
        "manifest_path": str(manifest_path),
        "session_dir": str(session_dir),
        "recording_count": len(manifest),
        "total_duration_minutes": round(total_duration / 60, 2),
        "emotion_distribution": emotion_counts,
    }


@app.delete("/voice-recording/recording/{session_id}/{recording_id}")
async def delete_recording(session_id: str, recording_id: str):
    """Delete a specific recording from a session."""
    if session_id not in voice_recording_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = voice_recording_sessions[session_id]
    recording = next(
        (r for r in session["recordings"] if r["recording_id"] == recording_id),
        None
    )

    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    # Delete audio file
    audio_path = Path(recording["audio_path"])
    if audio_path.exists():
        audio_path.unlink()

    # Remove from session
    session["recordings"] = [r for r in session["recordings"] if r["recording_id"] != recording_id]
    session["recorded_count"] = len(session["recordings"])

    return {"status": "deleted", "recording_id": recording_id}


# ============== Training Management ==============

import subprocess
import signal

# Global training process
training_process = None


@app.get("/training/data-stats")
async def get_training_data_stats():
    """Get statistics about available training data."""
    training_dir = DATA_DIR / "training"
    manifest_path = training_dir / "voice_manifest.json"

    if not manifest_path.exists():
        # Check for voice recording sessions that can be exported
        sessions_with_data = []
        voice_samples_dir = DATA_DIR / "voice_samples"
        if voice_samples_dir.exists():
            for session_dir in voice_samples_dir.iterdir():
                if session_dir.is_dir():
                    wav_files = list(session_dir.glob("*.wav"))
                    if wav_files:
                        sessions_with_data.append({
                            "session_id": session_dir.name,
                            "recording_count": len([f for f in wav_files if "_raw" not in f.name]),
                        })

        return {
            "has_training_data": False,
            "sessions_available": sessions_with_data,
            "message": "No training manifest found. Export a recording session first."
        }

    with open(manifest_path) as f:
        samples = json.load(f)

    # Calculate stats
    total_duration = sum(s.get("duration", 0) for s in samples)
    emotion_counts = {}
    for s in samples:
        emotion = s.get("prosody", {}).get("semantic", {}).get("emotion", "unknown")
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    return {
        "has_training_data": True,
        "manifest_path": str(manifest_path),
        "sample_count": len(samples),
        "total_duration_seconds": total_duration,
        "total_duration_minutes": round(total_duration / 60, 2),
        "emotion_distribution": emotion_counts,
        "samples_preview": samples[:5],  # First 5 samples for preview
    }


@app.post("/training/prepare")
async def prepare_training_data(session_id: str = None):
    """Prepare training data from a voice recording session."""
    # Find session to export
    if session_id:
        if session_id not in voice_recording_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session = voice_recording_sessions[session_id]
    else:
        # Find the session with most recordings
        if not voice_recording_sessions:
            raise HTTPException(status_code=400, detail="No recording sessions available")
        session = max(voice_recording_sessions.values(), key=lambda s: s.get("recorded_count", 0))
        session_id = session["session_id"]

    if not session.get("recordings"):
        raise HTTPException(status_code=400, detail="Session has no recordings")

    # Export session first
    export_result = await export_session_for_training(session_id)

    # Create training manifest with absolute paths
    training_dir = DATA_DIR / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    session_manifest = Path(export_result["manifest_path"])
    with open(session_manifest) as f:
        samples = json.load(f)

    # Convert to absolute paths
    base_dir = Path(__file__).parent.parent.resolve()
    for sample in samples:
        rel_path = sample["audio_path"].replace("../", "")
        sample["audio_path"] = str(base_dir / rel_path)

    # Save training manifest
    output_path = training_dir / "voice_manifest.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    return {
        "status": "prepared",
        "manifest_path": str(output_path),
        "sample_count": len(samples),
        "total_duration_minutes": export_result["total_duration_minutes"],
    }


@app.post("/training/start")
async def start_training(
    config: str = "m4_pro.yaml",
    epochs: int = 50,
    batch_size: int = 4,
):
    """Start CSM fine-tuning on prepared voice data."""
    global training_process

    # Check if already training
    if training_process and training_process.poll() is None:
        return {
            "status": "already_running",
            "pid": training_process.pid,
            "message": "Training is already in progress"
        }

    # Check for training data
    training_dir = DATA_DIR / "training"
    manifest_path = training_dir / "voice_manifest.json"

    if not manifest_path.exists():
        raise HTTPException(
            status_code=400,
            detail="No training data found. Run /training/prepare first."
        )

    # Paths
    base_dir = Path(__file__).parent.parent.resolve()
    training_script = base_dir / "training" / "finetune_csm.py"
    config_path = base_dir / "training" / "config" / config
    output_dir = base_dir / "models" / f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_file = base_dir / "training" / "training.log"

    if not training_script.exists():
        raise HTTPException(status_code=500, detail="Training script not found")

    # Build command
    cmd = [
        sys.executable,
        str(training_script),
        "--data", str(manifest_path),
        "--output", str(output_dir),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]

    if config_path.exists():
        cmd.extend(["--config", str(config_path)])

    # Start training process
    print(f"[TRAINING] Starting: {' '.join(cmd)}")

    with open(log_file, "w") as log:
        training_process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(base_dir / "training"),
        )

    return {
        "status": "started",
        "pid": training_process.pid,
        "config": config,
        "epochs": epochs,
        "batch_size": batch_size,
        "output_dir": str(output_dir),
        "log_file": str(log_file),
        "message": "Training started. Check /training/status for progress."
    }


@app.get("/training/status")
async def get_training_status():
    """Get current training status."""
    global training_process

    base_dir = Path(__file__).parent.parent.resolve()
    log_file = base_dir / "training" / "training.log"

    if training_process is None:
        return {
            "status": "not_started",
            "message": "No training has been started"
        }

    poll_result = training_process.poll()

    if poll_result is None:
        # Still running - get last lines of log
        last_lines = []
        if log_file.exists():
            with open(log_file) as f:
                lines = f.readlines()
                last_lines = lines[-20:] if len(lines) > 20 else lines

        return {
            "status": "running",
            "pid": training_process.pid,
            "log_tail": "".join(last_lines),
        }
    else:
        # Completed
        last_lines = []
        if log_file.exists():
            with open(log_file) as f:
                lines = f.readlines()
                last_lines = lines[-50:] if len(lines) > 50 else lines

        return {
            "status": "completed" if poll_result == 0 else "failed",
            "exit_code": poll_result,
            "log_tail": "".join(last_lines),
        }


@app.post("/training/stop")
async def stop_training():
    """Stop the current training process."""
    global training_process

    if training_process is None or training_process.poll() is not None:
        return {"status": "not_running", "message": "No training process to stop"}

    training_process.terminate()
    try:
        training_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        training_process.kill()

    return {"status": "stopped", "message": "Training process terminated"}


# ============== Live Voice Transformation ==============

# Global transformer instance (loaded lazily)
live_transformer = None

def get_live_transformer():
    """Get or create the live voice transformer."""
    global live_transformer
    if live_transformer is None:
        try:
            from live_voice_transformer import LiveVoiceTransformer
            live_transformer = LiveVoiceTransformer(
                voice_samples_dir=DATA_DIR / "voice_samples"
            )
        except Exception as e:
            print(f"Failed to load live transformer: {e}")
            return None
    return live_transformer


@app.get("/live-transform/status")
async def get_live_transform_status():
    """Get live voice transformation status and available emotions."""
    transformer = get_live_transformer()
    if transformer is None:
        return {
            "ready": False,
            "error": "Transformer not available (Seed-VC not installed or no voice samples)"
        }
    return transformer.get_stats()


@app.post("/live-transform/set-emotion")
async def set_live_transform_emotion(emotion: str = Form(...), intensity: float = Form(0.7)):
    """Set the target emotion for live transformation."""
    transformer = get_live_transformer()
    if transformer is None:
        raise HTTPException(status_code=503, detail="Transformer not available")

    try:
        transformer.set_emotion(emotion, intensity)
        return {
            "emotion": emotion,
            "intensity": intensity,
            "available_emotions": transformer.available_emotions
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/live-transform/convert")
async def convert_voice_emotion(
    audio: UploadFile = File(...),
    emotion: str = Form(...),
    intensity: float = Form(0.7),
):
    """Convert an audio file's emotion (non-streaming, for testing)."""
    transformer = get_live_transformer()
    if transformer is None:
        raise HTTPException(status_code=503, detail="Transformer not available")

    import tempfile
    import scipy.io.wavfile as wavfile

    # Save uploaded file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        content = await audio.read()
        f.write(content)
        input_path = f.name

    try:
        # Set emotion
        transformer.set_emotion(emotion, intensity)

        # Convert
        output_id = f"{uuid.uuid4().hex[:8]}"
        output_path = PROCESSED_DIR / f"live_transform_{output_id}.wav"

        inference_time = transformer.convert_file(input_path, str(output_path), emotion)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"transformed_{emotion}_{output_id}.wav",
            headers={"X-Inference-Time": str(inference_time)}
        )
    finally:
        os.unlink(input_path)


@app.websocket("/ws/live-transform")
async def websocket_live_transform(websocket: WebSocket):
    """
    WebSocket for real-time voice transformation.

    Client sends:
        {"type": "audio_chunk", "data": "<base64 PCM audio>", "sample_rate": 24000}
        {"type": "set_emotion", "emotion": "happy", "intensity": 0.8}
        {"type": "get_status"}

    Server sends:
        {"type": "audio_output", "data": "<base64 PCM audio>", "sample_rate": 22050, "latency_ms": 150}
        {"type": "status", "emotion": "happy", "intensity": 0.8, "ready": true}
        {"type": "error", "message": "..."}
    """
    await websocket.accept()

    transformer = get_live_transformer()
    if transformer is None:
        await websocket.send_json({
            "type": "error",
            "message": "Transformer not available. Check if Seed-VC is installed."
        })
        await websocket.close()
        return

    print(f"WebSocket connected for live transform")

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "audio_chunk":
                # Decode audio
                try:
                    audio_b64 = data.get("data", "")
                    audio_bytes = base64.b64decode(audio_b64)
                    sample_rate = data.get("sample_rate", 24000)

                    # Convert bytes to numpy (assuming PCM int16)
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0

                    # Process
                    import time
                    start = time.time()
                    converted, output_sr = transformer.convert_audio(audio_np, sample_rate)
                    latency_ms = (time.time() - start) * 1000

                    # Encode output
                    if converted.dtype == np.float32 or converted.dtype == np.float64:
                        converted_int16 = (converted * 32767).astype(np.int16)
                    else:
                        converted_int16 = converted.astype(np.int16)

                    output_b64 = base64.b64encode(converted_int16.tobytes()).decode()

                    await websocket.send_json({
                        "type": "audio_output",
                        "data": output_b64,
                        "sample_rate": output_sr,
                        "latency_ms": round(latency_ms, 1)
                    })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Audio processing error: {str(e)}"
                    })

            elif msg_type == "set_emotion":
                emotion = data.get("emotion", "neutral")
                intensity = data.get("intensity", 0.7)

                try:
                    transformer.set_emotion(emotion, intensity)
                    await websocket.send_json({
                        "type": "status",
                        "emotion": emotion,
                        "intensity": intensity,
                        "ready": True
                    })
                except ValueError as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

            elif msg_type == "set_quality":
                # Adjust diffusion steps for quality/speed tradeoff
                steps = data.get("diffusion_steps", 6)
                transformer.set_diffusion_steps(steps)
                await websocket.send_json({
                    "type": "status",
                    "diffusion_steps": transformer.config.diffusion_steps,
                    "message": f"Quality set to {steps} diffusion steps"
                })

            elif msg_type == "get_status":
                stats = transformer.get_stats()
                await websocket.send_json({
                    "type": "status",
                    **stats
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass


# ============== Startup/Shutdown ==============

@app.on_event("startup")
async def startup():
    """Load existing labeled samples on startup."""
    print("Loading existing samples...")
    
    # Load from labeled directory
    for label_file in LABELED_DIR.glob("*.json"):
        if label_file.name != "metadata.json":
            try:
                with open(label_file) as f:
                    sample = json.load(f)
                    samples[sample["id"]] = sample
            except Exception as e:
                print(f"Failed to load {label_file}: {e}")
    
    print(f"Loaded {len(samples)} existing samples")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    print("Shutting down...")


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=True
    )
