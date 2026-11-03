"""Output publication is guarded separately from reusable cancellation."""

import numpy as np
import pytest

from src.pipeline.response_handler import ResponseHandler
from tests.mocks.application_graph import OfflineTTS


@pytest.mark.asyncio
async def test_cancel_preserves_earcon_then_terminal_close_detaches_without_mutation():
    handler = ResponseHandler(OfflineTTS())
    audio = np.full(512, .125, dtype=np.float32)
    expected = audio.copy()
    try:
        handler.set_earcon(audio, 24000)
        await handler.cancel()
        assert handler._earcon_audio is audio
        assert not handler.get_delivery_status()['closed']
        np.testing.assert_array_equal(audio, expected)
        await handler.aclose()
        assert handler._earcon_audio is None
        np.testing.assert_array_equal(audio, expected)
        with pytest.raises(RuntimeError, match='^voice_delivery_closed$'):
            handler.set_earcon(audio, 24000)
        assert handler._earcon_audio is None
    finally:
        await handler.aclose()


@pytest.mark.asyncio
async def test_borrowed_attachment_cannot_publish_after_close_or_claim_ownership():
    class Borrowed:
        def close(self):
            raise AssertionError('unreviewed borrowed close must not be called')

    cache = Borrowed()
    handler = ResponseHandler(OfflineTTS())
    try:
        handler.attach_response_cache(cache)
        assert handler._response_cache is cache
        await handler.cancel()
        assert handler._response_cache is cache
        with pytest.raises(RuntimeError, match='^response_cache_unregistered$'):
            handler.attach_response_cache(cache, owned=True)
        assert handler._response_cache is cache
        await handler.aclose()
        assert handler._response_cache is None
        with pytest.raises(RuntimeError, match='^voice_delivery_closed$'):
            handler.attach_response_cache(cache)
        assert handler._response_cache is None
    finally:
        await handler.aclose()
