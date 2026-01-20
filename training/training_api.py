"""
Voice Clone Pipeline - Training Dashboard API
Standalone FastAPI server for training metrics and WebSocket updates.

Usage:
    python training_api.py --port 8001
    
    # Or run with training:
    python train_deepseek.py --config config/m4_pro_deepseek.yaml --dashboard
"""

import argparse
import asyncio
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn


# ============== Metrics Data Structure ==============

@dataclass
class TrainingMetrics:
    """Training metrics for dashboard."""
    step: int = 0
    epoch: int = 0
    epoch_progress: float = 0.0
    train_loss: float = 0.0
    val_loss: float = 0.0
    mtp_loss: float = 0.0
    learning_rate: float = 0.0
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    memory_used_gb: float = 0.0
    memory_peak_gb: float = 0.0
    memory_allocated_gb: float = 0.0
    grad_norm: float = 0.0
    grad_norm_clipped: bool = False
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    loss_history: List[Dict[str, float]] = field(default_factory=list)
    lr_history: List[Dict[str, float]] = field(default_factory=list)
    memory_history: List[Dict[str, float]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: str = "initializing"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsManager:
    """Thread-safe metrics manager with WebSocket broadcasting."""
    
    def __init__(self, max_history: int = 1000):
        self.metrics = TrainingMetrics()
        self.max_history = max_history
        self.lock = threading.Lock()
        self.websockets: List[WebSocket] = []
        self.update_queue = asyncio.Queue()
    
    def update(self, **kwargs):
        """Update metrics (called from training thread)."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)
            
            # Update histories
            if 'train_loss' in kwargs:
                self.metrics.loss_history.append({
                    'step': self.metrics.step,
                    'train_loss': self.metrics.train_loss,
                    'val_loss': self.metrics.val_loss,
                })
                if len(self.metrics.loss_history) > self.max_history:
                    self.metrics.loss_history.pop(0)
            
            if 'learning_rate' in kwargs:
                self.metrics.lr_history.append({
                    'step': self.metrics.step,
                    'lr': self.metrics.learning_rate,
                })
                if len(self.metrics.lr_history) > self.max_history:
                    self.metrics.lr_history.pop(0)
            
            if 'memory_used_gb' in kwargs:
                self.metrics.memory_history.append({
                    'step': self.metrics.step,
                    'used': self.metrics.memory_used_gb,
                    'peak': self.metrics.memory_peak_gb,
                })
                if len(self.metrics.memory_history) > self.max_history:
                    self.metrics.memory_history.pop(0)
    
    def add_error(self, error: str):
        """Add an error message."""
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.metrics.errors.append(f"[{timestamp}] {error}")
            self.metrics.status = "error"
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.metrics.warnings.append(f"[{timestamp}] {warning}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self.lock:
            return self.metrics.to_dict()
    
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast to all connected WebSockets."""
        dead_sockets = []
        for ws in self.websockets:
            try:
                await ws.send_json(data)
            except Exception:
                dead_sockets.append(ws)
        
        for ws in dead_sockets:
            self.websockets.remove(ws)


# Global metrics manager
metrics_manager = MetricsManager()


# ============== FastAPI App ==============

app = FastAPI(
    title="Voice Clone Training API",
    description="Real-time training metrics and monitoring",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API status."""
    return {
        "status": "ok",
        "service": "Voice Clone Training API",
        "metrics_endpoint": "/metrics",
        "websocket_endpoint": "/ws",
    }


@app.get("/metrics")
async def get_metrics():
    """Get current training metrics."""
    return metrics_manager.get_metrics()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time metrics updates."""
    await websocket.accept()
    metrics_manager.websockets.append(websocket)
    
    try:
        # Send initial state
        await websocket.send_json(metrics_manager.get_metrics())
        
        # Keep connection alive and send updates
        while True:
            # Wait for updates or periodic send
            await asyncio.sleep(0.5)
            await websocket.send_json(metrics_manager.get_metrics())
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in metrics_manager.websockets:
            metrics_manager.websockets.remove(websocket)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve embedded dashboard."""
    html_path = Path(__file__).parent.parent / "training-dashboard-demo.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Dashboard not found</h1><p>Run from project root.</p>")


# ============== Simulation for Testing ==============

async def simulate_training():
    """Simulate training progress for testing the dashboard."""
    import math
    
    step = 0
    start_time = time.time()
    
    while True:
        step += 1
        epoch = (step // 100) + 1
        epoch_progress = (step % 100) / 100
        elapsed = time.time() - start_time
        
        # Simulated metrics
        train_loss = 0.5 * math.exp(-step / 500) + 0.1 + (0.02 * math.sin(step / 20))
        val_loss = 0.5 * math.exp(-step / 500) + 0.12 + (0.03 * math.sin(step / 25))
        
        # LR with warmup and decay
        if step < 50:
            lr = 2e-5 * (step / 50)
        else:
            progress = (step - 50) / 950
            lr = 2e-5 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        
        metrics_manager.update(
            step=step,
            epoch=epoch,
            epoch_progress=epoch_progress,
            train_loss=train_loss,
            val_loss=val_loss,
            mtp_loss=train_loss * 0.3,
            learning_rate=lr,
            samples_per_second=2.5 + 0.5 * math.sin(step / 30),
            memory_used_gb=25 + 3 * math.sin(step / 40),
            memory_peak_gb=32,
            grad_norm=0.5 + 0.2 * abs(math.sin(step / 15)),
            elapsed_seconds=elapsed,
            eta_seconds=max(0, (5000 - step) * 0.5),
            status="training" if step < 5000 else "complete",
        )
        
        # Broadcast to WebSockets
        await metrics_manager.broadcast(metrics_manager.get_metrics())
        
        await asyncio.sleep(0.5)
        
        if step >= 5000:
            break


@app.on_event("startup")
async def startup():
    """Start background tasks."""
    # Optionally start simulation for demo
    # asyncio.create_task(simulate_training())
    pass


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Training Dashboard API")
    parser.add_argument("--port", type=int, default=8001, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--simulate", action="store_true", help="Run with simulated training data")
    
    args = parser.parse_args()
    
    if args.simulate:
        @app.on_event("startup")
        async def start_simulation():
            asyncio.create_task(simulate_training())
    
    print(f"Starting Training Dashboard API on http://{args.host}:{args.port}")
    print(f"  Metrics: http://localhost:{args.port}/metrics")
    print(f"  WebSocket: ws://localhost:{args.port}/ws")
    print(f"  Dashboard: http://localhost:{args.port}/dashboard")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
