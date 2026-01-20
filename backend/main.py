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

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torchaudio
import whisper

from prosody_analyzer import CompleteProsodyAnalyzer, ProsodyResult

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
