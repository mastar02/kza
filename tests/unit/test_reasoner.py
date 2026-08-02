"""
Tests for LLM Reasoner and FastRouter.
"""

from tests.mocks.mock_llm import MockFastRouter


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


class TestFastRouterLoRA:
    """Qué garantiza FastRouter sobre LoRA tras migrar a cliente HTTP.

    Ojo con ``test_fast_router_load_unload_lora`` de abajo: corre contra
    ``MockFastRouter``, así que verifica el mock, no producción. Queda en
    verde por eso — no porque el hot-swap funcione (ver
    ``test_lora_hot_swap_is_not_supported_over_http``).
    """

    def test_legacy_lora_kwargs_do_not_break_construction(self):
        """FastRouter absorbe los kwargs de la era vLLM sin romper.

        Reemplaza a ``test_fast_router_lora_init``, que asertaba
        ``router.enable_lora`` / ``router.max_lora_rank``. FastRouter migró a
        cliente HTTP (:8101) en el reorg de endpoints 2026-05-07 (#6) y esos
        atributos dejaron de existir: el test fallaba con AttributeError desde
        entonces, invisible porque la suite no corría en CI.

        Lo que producción SÍ garantiza hoy es que una config vieja que todavía
        pase esos kwargs no revienta el arranque — se absorben en
        ``**_ignored``. Este test fija la garantía vigente, no la borrada.
        """
        from src.llm.reasoner import FastRouter

        router = FastRouter(
            model="test",
            enable_lora=True,
            lora_path="/models/test_adapter",
            max_lora_rank=32,
        )

        # El objeto queda usable...
        assert router.model_name == "test"
        assert router.base_url
        # ...y los kwargs legacy NO se convierten en estado silencioso.
        assert not hasattr(router, "enable_lora")
        assert not hasattr(router, "max_lora_rank")

    def test_legacy_kwargs_are_logged_not_swallowed(self, caplog):
        """Absorberlos no es esconderlos: tienen que quedar en el log.

        Si se absorbieran en silencio, una config obsoleta sería indetectable:
        el operador cree que el LoRA está activo y no lo está.
        """
        import logging

        from src.llm.reasoner import FastRouter

        with caplog.at_level(logging.DEBUG, logger="src.llm.reasoner"):
            FastRouter(model="test", enable_lora=True, lora_path="/x", max_lora_rank=8)

        assert "ignorando kwargs legacy" in caplog.text
        assert "enable_lora" in caplog.text

    def test_fast_router_load_unload_lora(self):
        """Test LoRA hot-swap via load/unload"""
        router = MockFastRouter(enable_lora=True)

        assert router._lora_active is False

        router.load_lora("/models/nightly/latest")
        assert router._lora_active is True
        assert router._lora_path == "/models/nightly/latest"

        router.unload_lora()
        assert router._lora_active is False

    def test_lora_hot_swap_is_not_supported_over_http(self, caplog):
        """Reemplaza a ``test_fast_router_generate_with_lora``.

        Ese test parcheaba ``vllm.lora.request.LoRARequest`` en ``sys.modules``
        y verificaba que se pasara un LoRARequest a ``generate()`` — camino que
        murió con el cliente HTTP. Además tardaba 3.3s intentando conectar de
        verdad a :8101 una vez que ``openai`` estuvo instalado: un test
        unitario haciendo I/O de red.

        Hoy el adapter lo carga el service externo, no el cliente, y
        ``load_lora`` avisa exactamente eso. Fijar ESE contrato importa: si
        alguien vuelve a llamarlo esperando hot-swap, el warning es la única
        señal de que no pasó nada.
        """
        import logging

        from src.llm.reasoner import HttpReasoner

        reasoner = HttpReasoner(base_url="http://127.0.0.1:8101/v1")

        with caplog.at_level(logging.WARNING, logger="src.llm.reasoner"):
            reasoner.load_lora("/models/nightly/latest")

        assert "no carga LoRA" in caplog.text
