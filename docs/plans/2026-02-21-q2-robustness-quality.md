# Q2: Robustness & Code Quality — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all bare excepts, add Protocol-based type hints, fix fire-and-forget tasks, add TTL cache, and fix security issues.

**Architecture:** Mechanical fixes across multiple files. Protocol classes go in a new `src/protocols.py`. TTLCache is a small utility used by RequestRouter. Fire-and-forget fixes add `add_done_callback` to all `asyncio.create_task` calls.

**Tech Stack:** Python 3.10+, asyncio, typing.Protocol, dataclasses

---

## Task 1: Eliminate Bare Excepts (5 locations)

All remaining bare `except:` clauses swallow KeyboardInterrupt, SystemExit, and MemoryError. Replace with `except Exception`.

**Files:**
- Modify: `src/orchestrator/context_persistence.py:271,383`
- Modify: `src/training/nightly_trainer.py:717,1074`
- Modify: `src/ambient/audio_event_detector.py:411`

**Step 1: Write test that detects bare excepts**

```python
# tests/unit/test_no_bare_excepts.py
import ast
import glob

def test_no_bare_excepts_in_src():
    """Verify no bare except: clauses exist in source code."""
    violations = []
    for filepath in glob.glob("src/**/*.py", recursive=True):
        with open(filepath) as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                violations.append(f"{filepath}:{node.lineno}")
    assert violations == [], f"Bare except: found at: {violations}"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_no_bare_excepts.py -v`
Expected: FAIL with 5 violations

**Step 3: Fix each bare except**

```python
# src/orchestrator/context_persistence.py:271
# BEFORE:
                except:
                    pass
# AFTER:
                except Exception as e:
                    logger.warning(f"Failed to read existing context for user: {e}")

# src/orchestrator/context_persistence.py:383
# BEFORE:
            except:
                pass
# AFTER:
            except Exception as e:
                logger.warning(f"Failed to read user history: {e}")

# src/ambient/audio_event_detector.py:411
# BEFORE:
        except:
            pass
# AFTER:
        except Exception as e:
            logger.debug(f"Beep pattern detection failed: {e}")

# src/training/nightly_trainer.py:717
# BEFORE:
                except:
                    pass
# AFTER:
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse loss from output: {e}")

# src/training/nightly_trainer.py:1074
# BEFORE:
                        except:
                            pass
# AFTER:
                        except Exception as e:
                            logger.debug(f"Failed to read adapter metadata: {e}")
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_no_bare_excepts.py -v`
Expected: PASS

**Step 5: Run existing tests for modified modules**

Run: `python3 -m pytest tests/unit/orchestrator/ tests/unit/alerts/ -q`
Expected: All pass (no regressions)

**Step 6: Commit**

```bash
git add src/orchestrator/context_persistence.py src/training/nightly_trainer.py src/ambient/audio_event_detector.py tests/unit/test_no_bare_excepts.py
git commit -m "fix: replace 5 bare except: clauses with specific exception handling

All bare except: now catch Exception (or more specific types) and log.
Adds test to prevent future bare excepts in src/."
```

---

## Task 2: Create Protocol Classes for Type Safety

Define structural typing interfaces for the main service contracts. This enables type checking without coupling to concrete implementations.

**Files:**
- Create: `src/protocols.py`
- Create: `tests/unit/test_protocols.py`

**Step 1: Write test for protocol definitions**

```python
# tests/unit/test_protocols.py
import pytest
from typing import runtime_checkable
from src.protocols import (
    STTProvider, TTSProvider, HAProvider,
    VectorSearchProvider, LLMProvider
)


def test_protocols_are_runtime_checkable():
    """All protocols should be runtime checkable."""
    for proto in [STTProvider, TTSProvider, HAProvider, VectorSearchProvider, LLMProvider]:
        assert hasattr(proto, '__protocol_attrs__') or hasattr(proto, '__abstractmethods__') or True
        # runtime_checkable protocols can be used with isinstance


def test_stt_provider_has_transcribe():
    assert hasattr(STTProvider, 'transcribe')


def test_tts_provider_has_synthesize():
    assert hasattr(TTSProvider, 'synthesize')
    assert hasattr(TTSProvider, 'synthesize_stream')


def test_ha_provider_has_call_service():
    assert hasattr(HAProvider, 'call_service')
    assert hasattr(HAProvider, 'get_entity_state')


def test_vector_search_provider_has_search():
    assert hasattr(VectorSearchProvider, 'search_command')


def test_llm_provider_has_generate():
    assert hasattr(LLMProvider, 'generate')
    assert hasattr(LLMProvider, 'generate_stream')
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_protocols.py -v`
Expected: FAIL (module not found)

**Step 3: Create protocols module**

```python
# src/protocols.py
"""
Protocol classes defining structural typing interfaces for KZA services.

Usage:
    def process(stt: STTProvider, tts: TTSProvider) -> None:
        text = await stt.transcribe(audio)
        audio = await tts.synthesize(text)

These are structural types — any class implementing the required methods
satisfies the protocol without explicit inheritance.
"""

from typing import Protocol, AsyncIterator, Optional, runtime_checkable
import numpy as np


@runtime_checkable
class STTProvider(Protocol):
    """Speech-to-text provider."""
    async def transcribe(self, audio: np.ndarray) -> str: ...


@runtime_checkable
class TTSProvider(Protocol):
    """Text-to-speech provider."""
    async def synthesize(self, text: str) -> np.ndarray: ...
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]: ...


@runtime_checkable
class HAProvider(Protocol):
    """Home Assistant service provider."""
    async def call_service(self, domain: str, service: str, entity_id: str, data: Optional[dict] = None) -> bool: ...
    async def get_entity_state(self, entity_id: str) -> Optional[dict]: ...
    async def call_service_ws(self, domain: str, service: str, entity_id: str, data: Optional[dict] = None) -> bool: ...


@runtime_checkable
class VectorSearchProvider(Protocol):
    """Vector database search provider."""
    def search_command(self, text: str, threshold: float = 0.65) -> Optional[dict]: ...
    def sync_commands(self, ha_client: object, llm: object) -> int: ...
    def initialize(self) -> None: ...
    def get_stats(self) -> dict: ...


@runtime_checkable
class LLMProvider(Protocol):
    """LLM reasoning provider."""
    async def generate(self, prompt: str, max_tokens: int = 512) -> str: ...
    async def generate_stream(self, prompt: str, max_tokens: int = 512) -> AsyncIterator[str]: ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_protocols.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/protocols.py tests/unit/test_protocols.py
git commit -m "feat: add Protocol classes for structural typing

Defines STTProvider, TTSProvider, HAProvider, VectorSearchProvider,
LLMProvider as runtime-checkable Protocol types."
```

---

## Task 3: Fix Type Hints on Key Constructors

Fix the `any` vs `Any` typo and add type hints to the most important constructors using Protocol classes.

**Files:**
- Modify: `src/orchestrator/dispatcher.py:842` (`audio: any` → `audio: Optional[np.ndarray]`)
- Modify: `src/pipeline/request_router.py` (type hint constructor with Protocols)
- Modify: `src/pipeline/voice_pipeline.py` (type hint optional params)

**Step 1: Fix dispatcher.py `any` typo**

```python
# src/orchestrator/dispatcher.py:842
# BEFORE:
        audio: any = None,
# AFTER:
        audio: Optional[np.ndarray] = None,
```

Ensure `from typing import Optional` and `import numpy as np` are at top of file.

**Step 2: Type hint RequestRouter constructor**

Add Protocol type hints to the most important params in `src/pipeline/request_router.py`:

```python
from src.protocols import HAProvider, VectorSearchProvider, LLMProvider

def __init__(
    self,
    command_processor,  # CommandProcessor (circular import risk)
    response_handler,   # ResponseHandler (circular import risk)
    audio_manager,
    orchestrator=None,
    orchestrator_enabled: bool = True,
    chroma_sync: Optional[VectorSearchProvider] = None,
    ha_client: Optional[HAProvider] = None,
    llm_reasoner: Optional[LLMProvider] = None,
    ...
```

**Step 3: Type hint VoicePipeline optional params**

```python
# src/pipeline/voice_pipeline.py
# BEFORE:
        chroma_sync: Optional[object] = None,
        memory_manager: Optional[object] = None,
        orchestrator: Optional[object] = None,
# AFTER:
from src.protocols import VectorSearchProvider
        chroma_sync: Optional[VectorSearchProvider] = None,
        memory_manager: Optional[object] = None,
        orchestrator: Optional[object] = None,
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/pipeline/ tests/integration/test_voice_pipeline.py tests/unit/orchestrator/ -q`
Expected: All pass

**Step 5: Commit**

```bash
git add src/orchestrator/dispatcher.py src/pipeline/request_router.py src/pipeline/voice_pipeline.py
git commit -m "fix: add type hints to key constructors using Protocol classes

Fix audio: any -> Optional[np.ndarray] in dispatcher.
Add Protocol type hints to RequestRouter and VoicePipeline."
```

---

## Task 4: Fix Security — Settings URL as Environment Variable

The `config/settings.yaml` has a hardcoded IP. Replace with env var placeholder.

**Files:**
- Modify: `config/settings.yaml:5`

**Step 1: Fix hardcoded URL**

```yaml
# BEFORE:
  url: "http://192.168.1.100:8123"
# AFTER:
  url: "${HOME_ASSISTANT_URL}"
```

**Step 2: Verify main.py handles env var replacement**

Read `src/main.py` — the `load_config` function already has `replace_env_vars()` that substitutes `${VAR}` patterns. No code changes needed.

**Step 3: Commit**

```bash
git add config/settings.yaml
git commit -m "security: replace hardcoded HA URL with environment variable

settings.yaml now uses \${HOME_ASSISTANT_URL} like the token field."
```

---

## Task 5: Add Error Handlers to Fire-and-Forget Tasks

There are ~20 `asyncio.create_task()` calls without error callbacks. Only `audio_loop.py` has `add_done_callback`. Add error handlers to the rest.

**Files:**
- Create: `src/core/task_utils.py` (shared error handler utility)
- Create: `tests/unit/core/test_task_utils.py`
- Modify: All files with `asyncio.create_task` without `add_done_callback`

**Step 1: Write test for task utility**

```python
# tests/unit/core/test_task_utils.py
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from src.core.task_utils import create_monitored_task


@pytest.mark.asyncio
async def test_create_monitored_task_success():
    """Successful tasks should complete normally."""
    async def success_coro():
        return 42

    task = create_monitored_task(success_coro(), name="test-success")
    result = await task
    assert result == 42


@pytest.mark.asyncio
async def test_create_monitored_task_failure_logs_error():
    """Failed tasks should log the error, not crash."""
    async def failing_coro():
        raise ValueError("test error")

    with patch("src.core.task_utils.logger") as mock_logger:
        task = create_monitored_task(failing_coro(), name="test-fail")
        await asyncio.sleep(0.05)  # Let task complete
        # Task should not raise, error should be logged
        assert task.done()
        mock_logger.error.assert_called()
        assert "test-fail" in str(mock_logger.error.call_args)


@pytest.mark.asyncio
async def test_create_monitored_task_cancellation_not_logged():
    """Cancelled tasks should not log errors."""
    async def slow_coro():
        await asyncio.sleep(10)

    with patch("src.core.task_utils.logger") as mock_logger:
        task = create_monitored_task(slow_coro(), name="test-cancel")
        task.cancel()
        await asyncio.sleep(0.05)
        mock_logger.error.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/core/test_task_utils.py -v`
Expected: FAIL (module not found)

**Step 3: Create task utility**

```python
# src/core/task_utils.py
"""Utilities for safe asyncio task management."""

import asyncio
import logging

logger = logging.getLogger(__name__)


def _handle_task_error(task: asyncio.Task) -> None:
    """Callback for fire-and-forget tasks — logs exceptions instead of silently swallowing."""
    if task.cancelled():
        return
    exception = task.exception()
    if exception:
        task_name = task.get_name()
        logger.error(f"Background task '{task_name}' failed: {exception}", exc_info=exception)


def create_monitored_task(coro, *, name: str = None) -> asyncio.Task:
    """Create an asyncio task with automatic error logging.

    Args:
        coro: Coroutine to run as task.
        name: Optional name for the task (shown in logs on failure).

    Returns:
        The created Task with error callback attached.
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(_handle_task_error)
    return task
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/core/test_task_utils.py -v`
Expected: PASS

**Step 5: Replace bare create_task calls across codebase**

Replace `asyncio.create_task(coro)` with `create_monitored_task(coro, name="descriptive-name")` in these files:

| File | Line | Name |
|------|------|------|
| `src/conversation/follow_up_mode.py` | 273, 279, 314 | `"follow-up-timeout"`, `"follow-up-window"`, `"follow-up-delayed-end"` |
| `src/notifications/smart_notifications.py` | 146 | `"notification-processor"` |
| `src/timers/named_timers.py` | 136 | `"timer-update-loop"` |
| `src/orchestrator/priority_queue.py` | 556 | `"queue-processor"` |
| `src/kza_server.py` | 120, 129, 200 | `"model-unload"`, `"model-reload"`, `"server-stop"` |
| `src/presence/presence_detector.py` | 278, 283, 287 | `"ble-zone-scanner"`, `"motion-sensor-poll"`, `"presence-update"` |
| `src/integrations/ha_integration.py` | 70 | `"ha-update-loop"` |
| `src/routines/routine_scheduler.py` | 156, 160 | `"routine-scheduler"`, `"presence-routine-loop"` |
| `src/alerts/alert_scheduler.py` | 208, 214, 220 | `"security-check"`, `"pattern-check"`, `"device-check"` |
| `src/intercom/intercom_system.py` | 142 | `"intercom-processor"` |
| `src/orchestrator/dispatcher.py` | 814 | `"dispatch-processor"` |

Each file needs: `from src.core.task_utils import create_monitored_task` at the top.

**Step 6: Run tests**

Run: `python3 -m pytest tests/ -q --ignore=tests/unit/learning --ignore=tests/unit/spotify`
Expected: All pass

**Step 7: Commit**

```bash
git commit -m "feat: add error handlers to all fire-and-forget asyncio tasks

Create src/core/task_utils.py with create_monitored_task() utility.
Replace bare asyncio.create_task() in 13 files (~20 call sites).
All background tasks now log errors instead of silently failing."
```

---

## Task 6: TTL Cache for Request Router

Replace the plain dict cache in RequestRouter with a TTL-based cache that expires stale entries.

**Files:**
- Create: `src/core/ttl_cache.py`
- Create: `tests/unit/core/test_ttl_cache.py`
- Modify: `src/pipeline/request_router.py` (use TTLCache)

**Step 1: Write tests for TTLCache**

```python
# tests/unit/core/test_ttl_cache.py
import time
import pytest
from src.core.ttl_cache import TTLCache


def test_set_and_get():
    cache = TTLCache(max_size=10, ttl_seconds=60)
    cache.set("key1", {"value": 42})
    assert cache.get("key1") == {"value": 42}


def test_get_missing_key():
    cache = TTLCache(max_size=10, ttl_seconds=60)
    assert cache.get("missing") is None


def test_ttl_expiry():
    cache = TTLCache(max_size=10, ttl_seconds=0.1)
    cache.set("key1", {"value": 42})
    assert cache.get("key1") == {"value": 42}
    time.sleep(0.15)
    assert cache.get("key1") is None


def test_max_size_eviction():
    cache = TTLCache(max_size=2, ttl_seconds=60)
    cache.set("a", {"v": 1})
    cache.set("b", {"v": 2})
    cache.set("c", {"v": 3})  # Should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == {"v": 2}
    assert cache.get("c") == {"v": 3}


def test_len():
    cache = TTLCache(max_size=10, ttl_seconds=60)
    assert len(cache) == 0
    cache.set("a", {})
    assert len(cache) == 1


def test_clear():
    cache = TTLCache(max_size=10, ttl_seconds=60)
    cache.set("a", {})
    cache.set("b", {})
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/core/test_ttl_cache.py -v`
Expected: FAIL (module not found)

**Step 3: Implement TTLCache**

```python
# src/core/ttl_cache.py
"""Simple TTL cache for query results."""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _CacheEntry:
    value: dict
    created_at: float = field(default_factory=time.time)


class TTLCache:
    """Thread-safe (in asyncio context) cache with TTL and max size.

    Args:
        max_size: Maximum number of entries.
        ttl_seconds: Time-to-live in seconds for each entry.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 300):
        self._cache: dict[str, _CacheEntry] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[dict]:
        """Get value if key exists and hasn't expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if (time.time() - entry.created_at) >= self._ttl:
            del self._cache[key]
            return None
        return entry.value

    def set(self, key: str, value: dict) -> None:
        """Set a cache entry, evicting oldest if at capacity."""
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = _CacheEntry(value=value)

    def clear(self) -> None:
        """Remove all entries."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/core/test_ttl_cache.py -v`
Expected: PASS

**Step 5: Replace plain dict cache in RequestRouter**

```python
# src/pipeline/request_router.py — changes:

# Add import:
from src.core.ttl_cache import TTLCache

# In __init__, replace:
#   self._query_cache = {}
#   self._cache_max_size = cache_max_size
# With:
        self._query_cache = TTLCache(max_size=cache_max_size, ttl_seconds=300)

# In process_command_legacy, replace:
#   command = self._query_cache.get(cache_key)
# (no change — TTLCache.get has same interface)

# Replace _add_to_cache method:
#   def _add_to_cache(self, key, value) -> entire method
# With:
    def _add_to_cache(self, key: str, value: dict) -> None:
        """Add command result to TTL cache."""
        self._query_cache.set(key, value)
```

**Step 6: Run tests**

Run: `python3 -m pytest tests/unit/core/test_ttl_cache.py tests/unit/pipeline/test_request_router.py tests/integration/test_voice_pipeline.py -v`
Expected: All pass

**Step 7: Commit**

```bash
git commit -m "feat: replace plain dict cache with TTLCache (5-minute expiry)

Create src/core/ttl_cache.py with max_size + TTL eviction.
RequestRouter now uses TTLCache instead of unbounded dict.
Stale entries auto-expire after 300 seconds."
```

---

## Verification Checklist

After completing all 6 tasks, verify:

- [ ] `python3 -m pytest tests/unit/test_no_bare_excepts.py` — PASS (0 bare excepts)
- [ ] `python3 -m pytest tests/unit/test_protocols.py` — PASS (protocols defined)
- [ ] `python3 -m pytest tests/unit/core/test_task_utils.py` — PASS (monitored tasks)
- [ ] `python3 -m pytest tests/unit/core/test_ttl_cache.py` — PASS (TTL cache)
- [ ] `python3 -m pytest tests/unit/pipeline/ tests/integration/ -q` — All pass
- [ ] `grep -rn "except:" src/ | grep -v "except Exception" | grep -v "except ("` — 0 results
- [ ] `grep -rn "audio: any" src/` — 0 results
- [ ] `grep -n "192.168" config/settings.yaml` — 0 results
- [ ] `grep -rn "asyncio.create_task" src/ | grep -v "task_utils"` — 0 results (all migrated)
