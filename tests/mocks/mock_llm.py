"""
Mock LLM Reasoner for testing.
"""

from typing import Optional


class MockLLMReasoner:
    """Mock implementation of LLMReasoner for testing"""

    def __init__(self):
        self._loaded = False
        self._call_count = 0
        self._responses = {}  # prompt -> response mapping

    def load(self):
        self._loaded = True

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None
    ) -> dict:
        self._call_count += 1

        # Check for custom response
        for key, response in self._responses.items():
            if key in prompt:
                return {
                    "choices": [{"text": response}],
                    "usage": {"completion_tokens": len(response.split())}
                }

        # Default response
        return {
            "choices": [{"text": "Esta es una respuesta de prueba."}],
            "usage": {"completion_tokens": 10}
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        result = self(prompt, max_tokens, temperature)
        return result["choices"][0]["text"]

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        # Extract last user message
        user_msg = ""
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]

        return self.generate(user_msg, max_tokens, temperature)

    # Test helpers
    def set_response(self, prompt_contains: str, response: str):
        """Set a custom response for prompts containing a string"""
        self._responses[prompt_contains] = response

    def get_call_count(self) -> int:
        return self._call_count

    def reset(self):
        self._call_count = 0
        self._responses = {}


class MockFastRouter:
    """Mock implementation of FastRouter for testing"""

    def __init__(
        self,
        enable_lora: bool = False,
        lora_path: str = None,
        max_lora_rank: int = 32,
    ):
        self._loaded = False
        self._should_deep_reason = False
        self.enable_lora = enable_lora
        self.max_lora_rank = max_lora_rank
        self._lora_path = lora_path
        self._lora_active = False

    def load(self):
        self._loaded = True

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.3
    ) -> list[str]:
        return ["Respuesta rápida del router." for _ in prompts]

    def classify(self, text: str, options: list[str]) -> str:
        # Return first option by default
        return options[0] if options else "unknown"

    def should_use_deep_reasoning(self, text: str) -> bool:
        # Keywords that trigger deep reasoning
        complex_keywords = ["por qué", "explica", "cómo funciona", "diferencia"]
        for keyword in complex_keywords:
            if keyword in text.lower():
                return True
        return self._should_deep_reason

    def load_lora(self, lora_path: str):
        """Mock LoRA loading"""
        self._lora_path = lora_path
        self._lora_active = True

    def unload_lora(self):
        """Mock LoRA unloading"""
        self._lora_active = False

    # Test helpers
    def set_deep_reasoning(self, value: bool):
        self._should_deep_reason = value
