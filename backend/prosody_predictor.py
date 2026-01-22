"""
Prosody Predictor using Local LLM (DeepSeek/Qwen)

Analyzes text to automatically predict appropriate prosody for speech synthesis.
This enables natural generation without requiring the user to specify emotion.

Flow:
    "I'm so excited to see you!" → DeepSeek → {emotion: happy, energy: high, pace: fast}
                                                         ↓
                                               ProsodyEncoder → CSM → Audio

This replaces manual emotion selection with intelligent prediction.
"""

import json
import re
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

import torch


@dataclass
class PredictedProsody:
    """Predicted prosody values from text analysis."""
    emotion: str = "neutral"
    emotion_intensity: float = 0.5
    energy: float = 0.5          # 0-1, how energetic/dynamic
    pace: str = "medium"         # slow, medium, fast
    pitch_tendency: str = "neutral"  # low, neutral, high
    emphasis_words: list = None
    tone: str = "conversational"  # formal, casual, dramatic, etc.

    def __post_init__(self):
        if self.emphasis_words is None:
            self.emphasis_words = []


class ProsodyPredictor:
    """
    Predicts prosody from text using a local LLM.

    Supports:
    - DeepSeek models
    - Qwen models
    - OpenAI-compatible APIs (Ollama, vLLM)
    - Fallback rule-based prediction
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device: str = "auto",
        use_api: bool = False,
        api_base: str = "http://localhost:11434/v1",  # Ollama default
    ):
        self.model_name = model_name
        self.use_api = use_api
        self.api_base = api_base
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None

        if not use_api:
            self._load_model()

    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _load_model(self):
        """Load the local LLM for prosody prediction."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading prosody predictor: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
            )

            if self.device.type != "cuda":
                self.model = self.model.to(self.device)

            self.model.eval()
            print(f"Prosody predictor loaded on {self.device}")

        except Exception as e:
            print(f"Could not load LLM: {e}")
            print("Using rule-based fallback for prosody prediction")
            self.model = None

    def predict(self, text: str) -> PredictedProsody:
        """
        Predict prosody from text.

        Args:
            text: Input text to analyze

        Returns:
            PredictedProsody with emotion, energy, pace, etc.
        """
        # Try LLM prediction first
        if self.model is not None or self.use_api:
            try:
                return self._predict_with_llm(text)
            except Exception as e:
                print(f"LLM prediction failed: {e}, using fallback")

        # Fallback to rule-based
        return self._predict_rule_based(text)

    def _predict_with_llm(self, text: str) -> PredictedProsody:
        """Use LLM to predict prosody."""
        prompt = self._build_prompt(text)

        if self.use_api:
            response = self._call_api(prompt)
        else:
            response = self._generate_local(prompt)

        return self._parse_response(response, text)

    def _build_prompt(self, text: str) -> str:
        """Build the prompt for prosody prediction."""
        return f'''Analyze this text for speech synthesis prosody. Respond in JSON only.

Text: "{text}"

Predict these values:
- emotion: one of [neutral, happy, sad, angry, surprised, calm, excited, serious]
- emotion_intensity: 0.0 to 1.0 (how strong the emotion is)
- energy: 0.0 to 1.0 (overall energy level)
- pace: one of [slow, medium, fast]
- pitch_tendency: one of [low, neutral, high]
- emphasis_words: list of words that should be emphasized
- tone: one of [conversational, formal, casual, dramatic, sarcastic, sincere]

Respond with JSON only, no other text:
'''

    def _generate_local(self, prompt: str) -> str:
        """Generate with local model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part after the prompt
        response = response[len(prompt):].strip()
        return response

    def _call_api(self, prompt: str) -> str:
        """Call OpenAI-compatible API (Ollama, vLLM, etc.)."""
        import requests

        response = requests.post(
            f"{self.api_base}/chat/completions",
            json={
                "model": self.model_name.split("/")[-1],  # Just model name for Ollama
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200,
            },
            timeout=30,
        )

        if response.ok:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API error: {response.status_code}")

    def _parse_response(self, response: str, original_text: str) -> PredictedProsody:
        """Parse LLM response into PredictedProsody."""
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                return PredictedProsody(
                    emotion=data.get("emotion", "neutral"),
                    emotion_intensity=float(data.get("emotion_intensity", 0.5)),
                    energy=float(data.get("energy", 0.5)),
                    pace=data.get("pace", "medium"),
                    pitch_tendency=data.get("pitch_tendency", "neutral"),
                    emphasis_words=data.get("emphasis_words", []),
                    tone=data.get("tone", "conversational"),
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Failed to parse LLM response: {e}")

        # Fallback if parsing fails
        return self._predict_rule_based(original_text)

    def _predict_rule_based(self, text: str) -> PredictedProsody:
        """
        Rule-based prosody prediction as fallback.
        Analyzes punctuation, keywords, and patterns.
        """
        text_lower = text.lower()

        # Detect emotion from keywords
        emotion = "neutral"
        intensity = 0.5

        happy_words = ["happy", "excited", "great", "wonderful", "love", "amazing", "fantastic", "joy"]
        sad_words = ["sad", "sorry", "unfortunately", "regret", "miss", "lost", "disappointed"]
        angry_words = ["angry", "furious", "hate", "terrible", "awful", "unacceptable", "outraged"]
        surprised_words = ["wow", "amazing", "incredible", "unbelievable", "shocked", "surprised"]
        calm_words = ["please", "gently", "softly", "calmly", "peacefully", "relaxed"]

        if any(word in text_lower for word in happy_words):
            emotion = "happy"
            intensity = 0.7
        elif any(word in text_lower for word in sad_words):
            emotion = "sad"
            intensity = 0.6
        elif any(word in text_lower for word in angry_words):
            emotion = "angry"
            intensity = 0.8
        elif any(word in text_lower for word in surprised_words):
            emotion = "surprised"
            intensity = 0.7
        elif any(word in text_lower for word in calm_words):
            emotion = "calm"
            intensity = 0.5

        # Detect energy from punctuation
        exclamation_count = text.count("!")
        question_count = text.count("?")

        energy = 0.5
        if exclamation_count >= 2:
            energy = 0.9
            intensity = min(1.0, intensity + 0.2)
        elif exclamation_count == 1:
            energy = 0.7

        # Detect pace from text length and punctuation
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(1, len(words))

        if avg_word_len > 6 or len(words) > 20:
            pace = "slow"
        elif exclamation_count > 0 or emotion in ["excited", "angry", "surprised"]:
            pace = "fast"
        else:
            pace = "medium"

        # Pitch tendency
        if emotion in ["happy", "excited", "surprised"]:
            pitch = "high"
        elif emotion in ["sad", "serious"]:
            pitch = "low"
        else:
            pitch = "neutral"

        # Find words to emphasize (capitalized, repeated, or important)
        emphasis = []
        for word in words:
            if word.isupper() and len(word) > 2:
                emphasis.append(word.lower())
            elif word.endswith("!"):
                emphasis.append(word.rstrip("!").lower())

        # Detect tone
        if "?" in text and len(words) < 10:
            tone = "conversational"
        elif any(w in text_lower for w in ["please", "kindly", "would you"]):
            tone = "formal"
        elif exclamation_count > 1:
            tone = "dramatic"
        else:
            tone = "conversational"

        return PredictedProsody(
            emotion=emotion,
            emotion_intensity=intensity,
            energy=energy,
            pace=pace,
            pitch_tendency=pitch,
            emphasis_words=emphasis[:5],  # Max 5 words
            tone=tone,
        )

    def to_conditioning_dict(self, prosody: PredictedProsody) -> Dict[str, Any]:
        """
        Convert predicted prosody to conditioning format for ProsodyEncoder.

        Returns dict compatible with extract_prosody_for_conditioning format.
        """
        # Map emotion to semantic vector indices
        emotion_map = {
            "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
            "surprised": 4, "calm": 5, "excited": 1, "serious": 0
        }

        pace_map = {"slow": 0.3, "medium": 0.5, "fast": 0.7}
        pitch_map = {"low": 0.3, "neutral": 0.5, "high": 0.7}

        return {
            "semantic": {
                "emotions": {
                    prosody.emotion: prosody.emotion_intensity,
                },
                "emphasis_words": prosody.emphasis_words,
            },
            "acoustic": {
                "pitch_mean": pitch_map.get(prosody.pitch_tendency, 0.5) * 200,  # ~100-150 Hz
                "pitch_std": prosody.energy * 30,
                "hnr": 15 + prosody.energy * 10,
            },
            "rhythm": {
                "speaking_rate": pace_map.get(prosody.pace, 0.5) * 8,  # ~2-6 syllables/sec
                "pause_ratio": 0.4 - prosody.energy * 0.2,
            },
            "contour": {
                "pitch_trajectory": self._generate_contour(prosody),
            },
        }

    def _generate_contour(self, prosody: PredictedProsody) -> list:
        """Generate a pitch contour based on prosody."""
        import numpy as np

        length = 64
        t = np.linspace(0, 1, length)

        # Base contour depends on emotion
        if prosody.emotion in ["happy", "excited", "surprised"]:
            # Rising with variation
            contour = 0.5 + 0.2 * t + 0.1 * np.sin(t * 4 * np.pi)
        elif prosody.emotion in ["sad"]:
            # Falling
            contour = 0.6 - 0.2 * t
        elif prosody.emotion in ["angry"]:
            # High with spikes
            contour = 0.6 + 0.1 * np.random.randn(length)
        elif prosody.emotion in ["calm"]:
            # Flat and low
            contour = np.ones(length) * 0.4
        else:
            # Neutral - slight decline
            contour = 0.5 - 0.05 * t

        # Scale by intensity
        contour = contour * prosody.emotion_intensity + (1 - prosody.emotion_intensity) * 0.5

        return contour.tolist()


# Singleton instance for API use
_predictor: Optional[ProsodyPredictor] = None


def get_predictor(use_api: bool = True) -> ProsodyPredictor:
    """Get or create the global prosody predictor."""
    global _predictor
    if _predictor is None:
        # Default to rule-based (fast) for API use
        # Set use_api=True and configure for Ollama/vLLM if you have a local LLM
        _predictor = ProsodyPredictor(use_api=use_api)
    return _predictor


def predict_prosody(text: str) -> Dict[str, Any]:
    """
    Convenience function to predict prosody from text.

    Returns dict with prosody values ready for generation.
    """
    predictor = get_predictor(use_api=False)  # Use rule-based by default
    predicted = predictor.predict(text)
    return {
        "predicted": asdict(predicted),
        "conditioning": predictor.to_conditioning_dict(predicted),
    }


# Test
if __name__ == "__main__":
    predictor = ProsodyPredictor(use_api=False)  # Rule-based

    test_texts = [
        "Hello, how are you today?",
        "I'm SO EXCITED to see you!!!",
        "I'm really sorry to hear about your loss.",
        "This is absolutely UNACCEPTABLE!",
        "Please speak softly and calmly.",
        "Wow, that's incredible! I can't believe it!",
    ]

    print("Prosody Prediction Test")
    print("=" * 60)

    for text in test_texts:
        result = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"  Emotion: {result.emotion} ({result.emotion_intensity:.1f})")
        print(f"  Energy: {result.energy:.1f}, Pace: {result.pace}")
        print(f"  Pitch: {result.pitch_tendency}, Tone: {result.tone}")
        if result.emphasis_words:
            print(f"  Emphasis: {result.emphasis_words}")
