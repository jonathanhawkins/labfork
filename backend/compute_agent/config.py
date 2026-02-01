"""
Compute Agent Configuration

Configuration for the 4090 compute agent that connects to Workers for task dispatch.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DeviceConfig:
    """Device capabilities and identity."""

    name: str = "RTX-4090-Primary"
    platform: str = "cuda"  # cuda, metal, webgpu, cpu

    # Capabilities
    compute_tflops: float = 82.6  # RTX 4090 FP16 TFLOPS
    memory_gb: float = 24.0
    bandwidth_mbps: float = 1000.0

    # Supported models (can be updated at runtime)
    cached_models: List[str] = field(default_factory=lambda: [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ])


@dataclass
class WorkersConfig:
    """Cloudflare Workers connection settings."""

    # Workers API endpoint
    base_url: str = "https://labfork-agents.YOUR_SUBDOMAIN.workers.dev"

    # For local development
    local_url: str = "http://localhost:8787"

    # Use local for development
    use_local: bool = True

    @property
    def api_url(self) -> str:
        """Get the active API URL."""
        return self.local_url if self.use_local else self.base_url


@dataclass
class AgentConfig:
    """Agent behavior settings."""

    # Heartbeat interval (seconds)
    heartbeat_interval: int = 30

    # Task poll interval when idle (seconds)
    poll_interval: int = 5

    # Maximum concurrent tasks (usually 1 for GPU)
    max_concurrent_tasks: int = 1

    # Timeout for inference (seconds)
    inference_timeout: int = 300

    # Retry settings
    max_retries: int = 3
    retry_backoff: float = 1.0  # Exponential backoff base


@dataclass
class InferenceConfig:
    """Model inference settings."""

    # Default model
    default_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Model loading settings
    device: str = "cuda:0"
    torch_dtype: str = "bfloat16"  # or float16

    # Generation defaults
    default_max_tokens: int = 1024
    default_temperature: float = 0.7

    # vLLM settings (if using vLLM)
    use_vllm: bool = False
    vllm_tensor_parallel_size: int = 1


@dataclass
class Config:
    """Complete agent configuration."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = cls()

        # Override from environment
        if os.getenv("WORKERS_BASE_URL"):
            config.workers.base_url = os.getenv("WORKERS_BASE_URL")

        if os.getenv("WORKERS_LOCAL_URL"):
            config.workers.local_url = os.getenv("WORKERS_LOCAL_URL")

        if os.getenv("USE_LOCAL_WORKERS"):
            config.workers.use_local = os.getenv("USE_LOCAL_WORKERS", "true").lower() == "true"

        if os.getenv("DEVICE_NAME"):
            config.device.name = os.getenv("DEVICE_NAME")

        if os.getenv("CUDA_DEVICE"):
            config.inference.device = os.getenv("CUDA_DEVICE")

        if os.getenv("DEFAULT_MODEL"):
            config.inference.default_model = os.getenv("DEFAULT_MODEL")

        return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
