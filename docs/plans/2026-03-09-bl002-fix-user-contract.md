# BL-002: Fix Command/User Contract — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the untyped dict returned by `CommandProcessor.process_command()` with a typed `ProcessedCommand` dataclass, fixing the latent `AttributeError` when a speaker is enrolled.

**Architecture:** Create `ProcessedCommand` dataclass in `command_processor.py`. Update `_identify_speaker()` to return `User | None` directly. Update `RequestRouter` to consume typed attributes. Update all test helpers that mock the result.

**Tech Stack:** Python dataclasses, pytest, AsyncMock

---

### Task 1: Add `ProcessedCommand` dataclass and tests for it

**Files:**
- Modify: `src/pipeline/command_processor.py` (add dataclass at top, before class)
- Create: `tests/unit/pipeline/test_processed_command.py`

**Step 1: Write the test**

```python
"""Tests for ProcessedCommand dataclass."""
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

from src.pipeline.command_processor import ProcessedCommand


class TestProcessedCommand:
    def test_defaults(self):
        cmd = ProcessedCommand(text="hola")
        assert cmd.text == "hola"
        assert cmd.user is None
        assert cmd.emotion is None
        assert cmd.speaker_confidence == 0.0
        assert cmd.timings == {}
        assert cmd.success is False

    def test_with_user(self):
        mock_user = MagicMock()
        mock_user.user_id = "u1"
        mock_user.name = "Ana"
        cmd = ProcessedCommand(text="prende luz", user=mock_user, speaker_confidence=0.92, success=True)
        assert cmd.user.user_id == "u1"
        assert cmd.user.name == "Ana"
        assert cmd.speaker_confidence == 0.92

    def test_with_emotion(self):
        mock_emotion = MagicMock()
        mock_emotion.emotion = "happy"
        cmd = ProcessedCommand(text="que lindo dia", emotion=mock_emotion)
        assert cmd.emotion.emotion == "happy"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/pipeline/test_processed_command.py -v`
Expected: FAIL — `ImportError: cannot import name 'ProcessedCommand'`

**Step 3: Add the dataclass to `command_processor.py`**

Add after the existing imports (before `logger = logging.getLogger`), around line 10:

```python
from dataclasses import dataclass, field


@dataclass
class ProcessedCommand:
    """Typed result from CommandProcessor.process_command()."""
    text: str
    user: object | None = None  # User from user_manager or None
    emotion: object | None = None  # EmotionResult or None
    speaker_confidence: float = 0.0
    timings: dict = field(default_factory=dict)
    success: bool = False
```

Note: Use `object` for user/emotion types to avoid circular imports. The actual types are `User` and `EmotionResult` but importing them here would create a dependency cycle.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/pipeline/test_processed_command.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/pipeline/command_processor.py tests/unit/pipeline/test_processed_command.py
git commit -m "feat(BL-002): add ProcessedCommand dataclass

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Update `_identify_speaker()` to return `User | None`

**Files:**
- Modify: `src/pipeline/command_processor.py:229-263` (`_identify_speaker` method)
- Modify: `src/pipeline/command_processor.py:78-143` (`process_command` method)

**Step 1: Write the test**

Add to `tests/unit/pipeline/test_processed_command.py`:

```python
import numpy as np
from unittest.mock import AsyncMock

from src.pipeline.command_processor import CommandProcessor, ProcessedCommand


class TestCommandProcessorReturnsProcessedCommand:
    @pytest.mark.asyncio
    async def test_returns_processed_command_type(self):
        stt = MagicMock()
        stt.transcribe = MagicMock(return_value=("hola mundo", 50.0))
        cp = CommandProcessor(stt=stt)
        audio = np.zeros(16000, dtype=np.float32)

        result = await cp.process_command(audio, use_parallel=False)
        assert isinstance(result, ProcessedCommand)
        assert result.text == "hola mundo"
        assert result.success is True
        assert result.user is None

    @pytest.mark.asyncio
    async def test_with_identified_speaker(self):
        stt = MagicMock()
        stt.transcribe = MagicMock(return_value=("prende la luz", 50.0))

        mock_user = MagicMock()
        mock_user.user_id = "u1"
        mock_user.name = "Ana"

        mock_match = MagicMock()
        mock_match.is_known = True
        mock_match.user_id = "u1"
        mock_match.confidence = 0.91

        speaker_id = MagicMock()
        speaker_id.identify = MagicMock(return_value=mock_match)

        user_manager = MagicMock()
        user_manager.get_all_embeddings = MagicMock(return_value={"u1": np.zeros(192)})
        user_manager.get_user = MagicMock(return_value=mock_user)
        user_manager.update_last_seen = MagicMock()

        cp = CommandProcessor(
            stt=stt,
            speaker_identifier=speaker_id,
            user_manager=user_manager,
        )

        audio = np.zeros(16000, dtype=np.float32)
        result = await cp.process_command(audio, use_parallel=False)

        assert isinstance(result, ProcessedCommand)
        assert result.user is mock_user
        assert result.speaker_confidence == 0.91

    @pytest.mark.asyncio
    async def test_empty_text_is_not_success(self):
        stt = MagicMock()
        stt.transcribe = MagicMock(return_value=("   ", 50.0))
        cp = CommandProcessor(stt=stt)
        audio = np.zeros(16000, dtype=np.float32)

        result = await cp.process_command(audio, use_parallel=False)
        assert result.success is False
```

Add `import pytest` at the top of the file if not already present.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/pipeline/test_processed_command.py -v`
Expected: FAIL — `assert isinstance(result, ProcessedCommand)` fails because `process_command` still returns dict

**Step 3: Update `_identify_speaker()` and `process_command()`**

In `_identify_speaker()` (around line 229), change the return from:
```python
return {
    "user": user,
    "confidence": match.confidence,
    "timing_ms": timing_ms
}
```
to:
```python
return user, match.confidence, timing_ms
```

And change the `return None` at the end (line 263) to:
```python
return None, 0.0, 0.0
```

In `process_command()` (around line 78), change the method to build and return a `ProcessedCommand`:

Replace the full method body (lines 98-143) with:

```python
        result = ProcessedCommand(text="")
        pipeline_start = time.perf_counter()

        if use_parallel and (self.speaker_id or self.emotion_detector):
            text, stt_ms, speaker_result, emotion_result = await self._process_parallel(audio)
            result.timings["stt"] = stt_ms
            if speaker_result:
                user, confidence, spk_ms = speaker_result
                result.user = user
                result.speaker_confidence = confidence
                result.timings["speaker_id"] = spk_ms
            if emotion_result:
                result.emotion = emotion_result
                result.timings["emotion"] = emotion_result.processing_time_ms
        else:
            text, stt_ms = self.stt.transcribe(audio, self.sample_rate)
            result.timings["stt"] = stt_ms

            # Sequential speaker ID (fallback)
            if self.speaker_id and self.user_manager:
                user, confidence, spk_ms = self._identify_speaker(audio)
                result.user = user
                result.speaker_confidence = confidence
                result.timings["speaker_id"] = spk_ms

        result.text = text
        result.success = bool(text.strip())

        if result.success:
            self._current_user = result.user
            self._current_emotion = result.emotion

        result.timings["total"] = (time.perf_counter() - pipeline_start) * 1000

        logger.info(
            f"[CommandProcessor] Text='{text[:50]}' | "
            f"User={result.user.name if result.user else 'unknown'} | "
            f"Emotion={result.emotion.emotion if result.emotion else 'none'}"
        )

        return result
```

Also update `_process_parallel()` return type hint (line 145) from:
```python
async def _process_parallel(self, audio: np.ndarray) -> tuple[str, float, dict | None, object | None]:
```
to:
```python
async def _process_parallel(self, audio: np.ndarray) -> tuple[str, float, tuple | None, object | None]:
```

And update how `_process_parallel` wraps speaker_result (around line 182):
```python
        speaker_result = (
            results[1] if self.speaker_id and self.user_manager
            and not isinstance(results[1], Exception) and results[1] is not None
            else None
        )
```
This already works since `_identify_speaker` now returns a tuple — `results[1]` will be `(User, confidence, timing_ms)` or `(None, 0.0, 0.0)`. But we need to differentiate "no speaker ID configured" from "speaker not found". Update to:

```python
        if self.speaker_id and self.user_manager and not isinstance(results[1], Exception):
            speaker_result = results[1]  # tuple: (User|None, confidence, timing_ms)
        else:
            speaker_result = None
```

And in `process_command`, handle the tuple from parallel path:
The parallel path already sets `speaker_result` to the tuple. In the `if speaker_result:` block, we need to unpack properly. But `speaker_result` could be `(None, 0.0, 0.0)` which is truthy. Change the check:

```python
            if speaker_result:
                user, confidence, spk_ms = speaker_result
                if user is not None:
                    result.user = user
                    result.speaker_confidence = confidence
                result.timings["speaker_id"] = spk_ms
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/pipeline/test_processed_command.py -v`
Expected: All 6 tests pass

**Step 5: Commit**

```bash
git add src/pipeline/command_processor.py tests/unit/pipeline/test_processed_command.py
git commit -m "refactor(BL-002): CommandProcessor returns ProcessedCommand, _identify_speaker returns User directly

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Update `RequestRouter` to consume `ProcessedCommand`

**Files:**
- Modify: `src/pipeline/request_router.py:211-224` (orchestrated path)
- Modify: `src/pipeline/request_router.py:327-343` (legacy path)

**Step 1: Run existing router tests to establish baseline**

Run: `pytest tests/unit/pipeline/test_request_router.py tests/unit/pipeline/test_request_router_room.py -v 2>&1 | tail -20`
Expected: Note current pass/fail counts

**Step 2: Update `_process_command_orchestrated()` (around line 211)**

Replace lines 211-224:

Old:
```python
        cmd_result = await self.command_processor.process_command(audio, use_parallel=True)
        text = cmd_result["text"]
        result["text"] = text
        result["timings"].update(cmd_result["timings"])

        if not text.strip():
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 2. Get user info
        user = cmd_result.get("user")
        emotion = cmd_result.get("emotion")
        user_id = user.user_id if user else None
        user_name = user.name if user else None
```

New:
```python
        cmd = await self.command_processor.process_command(audio, use_parallel=True)
        text = cmd.text
        result["text"] = text
        result["timings"].update(cmd.timings)

        if not text.strip():
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 2. Get user info
        user = cmd.user
        emotion = cmd.emotion
        user_id = user.user_id if user else None
        user_name = user.name if user else None
```

Also update line 244-247 — this already accesses `user` correctly since `user` is now the `User` object directly:
```python
        if user:
            result["user"] = {
                "name": user_name,
                "permission_level": user.permission_level.name
            }
```
This is correct now — `user` is the `User` object.

**Step 3: Update `_process_command_legacy()` (around line 327)**

Replace lines 327-343:

Old:
```python
        cmd_result = await self.command_processor.process_command(audio, use_parallel=True)
        text = cmd_result["text"]
        result["text"] = text
        result["timings"].update(cmd_result["timings"])

        if not text.strip():
            return result

        user = cmd_result.get("user")
        emotion = cmd_result.get("emotion")
        user_id = user.user_id if user else None

        if user:
            result["user"] = {
                "name": user.name,
                "permission_level": user.permission_level.name
            }
```

New:
```python
        cmd = await self.command_processor.process_command(audio, use_parallel=True)
        text = cmd.text
        result["text"] = text
        result["timings"].update(cmd.timings)

        if not text.strip():
            return result

        user = cmd.user
        emotion = cmd.emotion
        user_id = user.user_id if user else None

        if user:
            result["user"] = {
                "name": user.name,
                "permission_level": user.permission_level.name
            }
```

**Step 4: Commit (tests will be fixed in next task)**

```bash
git add src/pipeline/request_router.py
git commit -m "refactor(BL-002): RequestRouter consumes ProcessedCommand typed attributes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Update test helpers to return `ProcessedCommand`

**Files:**
- Modify: `tests/unit/pipeline/test_request_router.py` (lines 35-42, `_make_cmd_result`)
- Modify: `tests/unit/pipeline/test_request_router_room.py` (lines 35-42, `_make_cmd_result`)
- Modify: `tests/integration/test_voice_pipeline.py` (lines 86-87, 112-113, etc.)

**Step 1: Update `_make_cmd_result` in `test_request_router.py`**

Replace `_make_cmd_result` (lines 35-42):

Old:
```python
def _make_cmd_result(text="enciende la luz", user=None, emotion=None, timings=None):
    """Create a mock CommandProcessor result."""
    return {
        "text": text,
        "user": user,
        "emotion": emotion,
        "timings": timings or {"stt": 50.0}
    }
```

New:
```python
from src.pipeline.command_processor import ProcessedCommand

def _make_cmd_result(text="enciende la luz", user=None, emotion=None, timings=None):
    """Create a mock CommandProcessor result."""
    return ProcessedCommand(
        text=text,
        user=user,
        emotion=emotion,
        timings=timings or {"stt": 50.0},
        success=bool(text.strip()),
    )
```

Add the import after the existing imports block (around line 28).

**Step 2: Same change in `test_request_router_room.py`**

Replace the same `_make_cmd_result` function with the identical `ProcessedCommand` version.
Add `from src.pipeline.command_processor import ProcessedCommand` to the imports.

**Step 3: Update `tests/integration/test_voice_pipeline.py`**

Each `process_command` mock return value (lines 86, 112, 127, 147, 162) needs to change from dict to `ProcessedCommand`. Example for line 86:

Old:
```python
pipeline.command_processor.process_command = AsyncMock(return_value={
    "text": "prende la luz del living",
    "user": None,
    "emotion": None,
    "timings": {"stt": 50.0, "total": 55.0}
})
```

New:
```python
from src.pipeline.command_processor import ProcessedCommand
# ...
pipeline.command_processor.process_command = AsyncMock(return_value=ProcessedCommand(
    text="prende la luz del living",
    timings={"stt": 50.0, "total": 55.0},
    success=True,
))
```

Apply this pattern to all 5 mock return values in that file. Add the import at top.

**Step 4: Run all affected tests**

Run: `pytest tests/unit/pipeline/test_request_router.py tests/unit/pipeline/test_request_router_room.py tests/unit/pipeline/test_processed_command.py tests/integration/test_voice_pipeline.py -v 2>&1 | tail -30`
Expected: All tests pass

**Step 5: Commit**

```bash
git add tests/unit/pipeline/test_request_router.py tests/unit/pipeline/test_request_router_room.py tests/integration/test_voice_pipeline.py
git commit -m "test(BL-002): update test helpers to use ProcessedCommand

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Update remaining consumers and verify full suite

**Files:**
- Modify: `tests/unit/pipeline/test_slim_pipeline.py` (line 153)
- Modify: `tests/unit/pipeline/test_voice_pipeline_room.py` (line 32)

**Step 1: Update `test_slim_pipeline.py`**

Line 153 has:
```python
request_router.process_command = AsyncMock(return_value={
    "text": "test", "success": True
})
```

This mocks `RequestRouter.process_command` (NOT `CommandProcessor.process_command`), so it's the router's OUTPUT dict — this does NOT need to change. Verify by reading the test.

**Step 2: Update `test_voice_pipeline_room.py`**

Line 32 has:
```python
process_command=AsyncMock(return_value={"text": "ok", "success": True})
```

Same situation — this mocks `request_router.process_command` (the router's return), not the `CommandProcessor`. Does NOT need to change.

**Step 3: Run the full pipeline test suite**

Run: `pytest tests/unit/pipeline/ tests/integration/test_voice_pipeline.py -v 2>&1 | tail -40`
Expected: All tests pass

**Step 4: Run broader test check for any other breakage**

Run: `pytest tests/ -x --timeout=30 2>&1 | tail -20`
Expected: No new failures (some tests may skip due to missing hardware)

**Step 5: Update backlog status**

In `docs/plans/2026-03-06-production-backlog.md`, change BL-002 row from `Proposed` to `Done`.

**Step 6: Commit**

```bash
git add docs/plans/2026-03-06-production-backlog.md
git commit -m "docs(BL-002): mark BL-002 as done — ProcessedCommand contract in place

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
