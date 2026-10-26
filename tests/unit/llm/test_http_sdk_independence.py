"""Cold async generation works with all SDK imports denied and GC enabled."""

import subprocess
import sys
import textwrap


def test_cold_async_paths_do_not_initialize_openai_sdk():
    code = textwrap.dedent('''
        import asyncio
        import gc
        import importlib.abc
        import sys
        from contextlib import aclosing

        class DenySDK(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'openai' or fullname.startswith('openai.'):
                    raise AssertionError('Async generation imported SDK types: ' + fullname)

        sys.meta_path.insert(0, DenySDK())
        from src.llm.reasoner import HttpReasoner
        from tests.mocks.llm_http import Endpoint, SSEBody, event

        async def check():
            endpoint = Endpoint(SSEBody([event('Respuesta completa.'), event(finish='stop')]))
            endpoint.completion_payload = {'choices': [{'message': {'content': 'Respuesta completa.'}, 'finish_reason': 'stop'}]}
            reasoner = HttpReasoner(model='configured-model', api_style='chat', cloud_consent=True)
            reasoner._new_async_client = endpoint.build
            async with aclosing(reasoner.astream('synthetic')) as stream:
                assert ''.join([chunk['token'] async for chunk in stream]) == 'Respuesta completa.'
            assert endpoint.body.closed and endpoint.client.is_closed()
            assert await reasoner.complete('synthetic') == 'Respuesta completa.'
            assert endpoint.client.is_closed()
            assert gc.isenabled()
            assert not any(name == 'openai' or name.startswith('openai.') for name in sys.modules)

        asyncio.run(check())
    ''')
    subprocess.run([sys.executable, '-c', code], check=True, timeout=20, capture_output=True, text=True)
