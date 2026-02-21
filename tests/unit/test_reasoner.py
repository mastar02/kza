"""
Tests for LLM Reasoner and FastRouter.
"""

import pytest
from tests.mocks.mock_llm import MockLLMReasoner, MockFastRouter


class TestLLMReasoner:
    """Test suite for LLM Reasoner"""

    def test_generate_response(self, mock_llm):
        """Test generating a text response"""
        response = mock_llm.generate("Test prompt")

        assert response is not None
        assert len(response) > 0

    def test_call_returns_dict(self, mock_llm):
        """Test __call__ returns proper structure"""
        result = mock_llm("Test prompt")

        assert "choices" in result
        assert "usage" in result
        assert len(result["choices"]) > 0
        assert "text" in result["choices"][0]

    def test_custom_response(self, mock_llm):
        """Test setting custom responses"""
        mock_llm.set_response("clima", "El clima está soleado hoy.")

        response = mock_llm.generate("¿Cómo está el clima?")
        assert "soleado" in response

    def test_chat_format(self, mock_llm):
        """Test chat-style interaction"""
        messages = [
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": "Hola"}
        ]

        response = mock_llm.chat(messages)
        assert response is not None

    def test_call_count_tracking(self, mock_llm):
        """Test that calls are counted"""
        mock_llm.reset()
        assert mock_llm.get_call_count() == 0

        mock_llm.generate("Test 1")
        mock_llm.generate("Test 2")
        mock_llm.generate("Test 3")

        assert mock_llm.get_call_count() == 3


class TestFastRouter:
    """Test suite for FastRouter"""

    def test_generate_batch(self):
        """Test batch generation"""
        router = MockFastRouter()

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = router.generate(prompts)

        assert len(responses) == 3
        assert all(isinstance(r, str) for r in responses)

    def test_classify(self):
        """Test classification"""
        router = MockFastRouter()

        options = ["domótica", "conversación", "rutina"]
        result = router.classify("prende la luz", options)

        assert result in options

    def test_should_use_deep_reasoning_simple(self):
        """Test detection of simple queries"""
        router = MockFastRouter()

        # Simple query
        assert router.should_use_deep_reasoning("prende la luz") is False

    def test_should_use_deep_reasoning_complex(self):
        """Test detection of complex queries"""
        router = MockFastRouter()

        # Complex queries with keywords
        assert router.should_use_deep_reasoning("¿por qué el cielo es azul?") is True
        assert router.should_use_deep_reasoning("explica la fotosíntesis") is True
        assert router.should_use_deep_reasoning("¿cómo funciona un motor?") is True

    def test_set_deep_reasoning_override(self):
        """Test overriding deep reasoning detection"""
        router = MockFastRouter()

        router.set_deep_reasoning(True)
        assert router.should_use_deep_reasoning("hola") is True

        router.set_deep_reasoning(False)
        # Still True because of keywords
        assert router.should_use_deep_reasoning("¿por qué?") is True
        # False for simple
        assert router.should_use_deep_reasoning("hola") is False
