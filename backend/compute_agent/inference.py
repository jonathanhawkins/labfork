"""
Inference Engine

Handles model loading and inference execution on the 4090 GPU.
This replaces Workers AI with local GPU inference.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import get_config, InferenceConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from model inference."""

    success: bool
    output: Optional[str] = None
    tokens_generated: int = 0
    compute_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "output": self.output,
            "metrics": {
                "computeTime": self.compute_time_ms,
                "tokensGenerated": self.tokens_generated,
                "tokensPerSecond": self.tokens_per_second,
            },
        }


class InferenceEngine:
    """
    Manages model loading and inference execution.

    Supports multiple models and handles loading/unloading as needed.
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or get_config().inference
        self.model = None
        self.tokenizer = None
        self.current_model_id: Optional[str] = None
        self.device = self.config.device

        # Model cache for quick switching
        self._model_cache: Dict[str, Any] = {}

        logger.info(f"InferenceEngine initialized, device={self.device}")

    def load_model(self, model_id: str) -> bool:
        """
        Load a model for inference.

        Args:
            model_id: HuggingFace model ID or local path

        Returns:
            True if model loaded successfully
        """
        if self.current_model_id == model_id and self.model is not None:
            logger.info(f"Model {model_id} already loaded")
            return True

        try:
            logger.info(f"Loading model: {model_id}")
            start_time = time.time()

            # Check cache first
            if model_id in self._model_cache:
                self.model, self.tokenizer = self._model_cache[model_id]
                self.current_model_id = model_id
                logger.info(f"Loaded {model_id} from cache")
                return True

            # Determine torch dtype
            torch_dtype = getattr(torch, self.config.torch_dtype, torch.bfloat16)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
            )

            self.current_model_id = model_id
            load_time = time.time() - start_time

            # Cache for future use (limit cache size)
            if len(self._model_cache) < 2:
                self._model_cache[model_id] = (self.model, self.tokenizer)

            logger.info(f"Model {model_id} loaded in {load_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False

    def unload_model(self):
        """Unload current model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.current_model_id = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded")

    def inference(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> InferenceResult:
        """
        Run inference on the loaded model.

        Args:
            prompt: Input prompt or user message
            model_id: Optional model to use (will load if needed)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            messages: Optional chat messages format

        Returns:
            InferenceResult with output and metrics
        """
        try:
            # Load model if needed
            target_model = model_id or self.config.default_model
            if not self.load_model(target_model):
                return InferenceResult(
                    success=False,
                    error=f"Failed to load model: {target_model}",
                )

            start_time = time.time()

            # Format input as chat messages if needed
            if messages:
                chat_messages = messages
            else:
                chat_messages = []
                if system_prompt:
                    chat_messages.append({"role": "system", "content": system_prompt})
                chat_messages.append({"role": "user", "content": prompt})

            # Apply chat template
            input_text = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.device)

            input_length = inputs.input_ids.shape[1]

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            # Decode output (only new tokens)
            output_tokens = outputs[0][input_length:]
            output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)

            # Calculate metrics
            compute_time_ms = (time.time() - start_time) * 1000
            tokens_generated = len(output_tokens)
            tokens_per_second = (tokens_generated / compute_time_ms) * 1000 if compute_time_ms > 0 else 0

            logger.info(
                f"Inference complete: {tokens_generated} tokens in {compute_time_ms:.0f}ms "
                f"({tokens_per_second:.1f} tok/s)"
            )

            return InferenceResult(
                success=True,
                output=output_text,
                tokens_generated=tokens_generated,
                compute_time_ms=compute_time_ms,
                tokens_per_second=tokens_per_second,
            )

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return InferenceResult(
                success=False,
                error=str(e),
            )

    def generate_embedding(
        self,
        text: str,
        model_id: Optional[str] = None,
    ) -> InferenceResult:
        """
        Generate text embedding.

        Uses the last hidden state of the model as embedding.
        """
        try:
            target_model = model_id or self.config.default_model
            if not self.load_model(target_model):
                return InferenceResult(
                    success=False,
                    error=f"Failed to load model: {target_model}",
                )

            start_time = time.time()

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Get embeddings (mean of last hidden state)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1].mean(dim=1)

            # Convert to list
            embedding_list = embeddings[0].cpu().tolist()

            compute_time_ms = (time.time() - start_time) * 1000

            return InferenceResult(
                success=True,
                output=str(embedding_list),  # Serialize as string
                compute_time_ms=compute_time_ms,
            )

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return InferenceResult(
                success=False,
                error=str(e),
            )

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            }

        return {
            "currentModel": self.current_model_id,
            "device": self.device,
            "gpuMemory": gpu_memory,
            "cachedModels": list(self._model_cache.keys()),
        }


# Global engine instance
_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get the global inference engine instance."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
