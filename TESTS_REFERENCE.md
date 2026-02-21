# Quick Reference: New Tests

## File Structure

```
/sessions/great-epic-planck/mnt/kza/
├── pytest.ini                                    # Pytest configuration
├── TESTING.md                                    # Detailed testing guide
├── TESTS_REFERENCE.md                           # This file
├── tests/unit/
│   ├── users/
│   │   ├── __init__.py
│   │   └── test_speaker_identifier.py           # 23 tests
│   ├── memory/
│   │   ├── __init__.py
│   │   └── test_memory_manager.py               # 38 tests
│   └── training/
│       ├── __init__.py
│       └── test_personality.py                  # 41 tests
└── src/
    ├── users/
    │   └── speaker_identifier.py                # (module under test)
    ├── memory/
    │   └── memory_manager.py                    # (module under test)
    └── training/
        └── personality.py                       # (module under test)
```

## Test Summary by Module

### 1. Speaker Identifier (23 tests)
**File**: `/sessions/great-epic-planck/mnt/kza/tests/unit/users/test_speaker_identifier.py`
**Source**: `/sessions/great-epic-planck/mnt/kza/src/users/speaker_identifier.py`

| Class | Tests | Purpose |
|-------|-------|---------|
| TestSpeakerIdentifierInit | 2 | Initialization with default/custom params |
| TestSpeakerIdentifierComputeSimilarity | 5 | Cosine similarity computation |
| TestSpeakerIdentifierGetEmbedding | 2 | Audio embedding extraction (mocked) |
| TestSpeakerIdentifierIdentify | 4 | Speaker identification logic |
| TestSpeakerIdentifierVerify | 2 | Speaker verification |
| TestSpeakerIdentifierCreateEnrollmentEmbedding | 3 | Enrollment from samples |
| TestSpeakerMatch | 2 | SpeakerMatch dataclass |
| TestSpeakerIdentifierEdgeCases | 3 | Edge cases and errors |

**Key Mocks**: `@patch` for GPU-intensive operations

### 2. Memory Manager (38 tests)
**File**: `/sessions/great-epic-planck/mnt/kza/tests/unit/memory/test_memory_manager.py`
**Source**: `/sessions/great-epic-planck/mnt/kza/src/memory/memory_manager.py`

| Class | Tests | Purpose |
|-------|-------|---------|
| TestConversationTurn | 2 | Conversation turn dataclass |
| TestUserPreference | 1 | User preference dataclass |
| TestMemoryFact | 1 | Memory fact dataclass |
| TestShortTermMemory | 8 | Short-term memory operations |
| TestLongTermMemory | 7 | Long-term memory with ChromaDB |
| TestPreferencesStore | 7 | JSON-based preferences |
| TestMemoryManager | 10 | Main coordinator |
| TestMemoryManagerIntegration | 1 | Full workflow integration |

**Key Mocks**: ChromaDB client, tempfile for I/O

### 3. Personality Manager (41 tests)
**File**: `/sessions/great-epic-planck/mnt/kza/tests/unit/training/test_personality.py`
**Source**: `/sessions/great-epic-planck/mnt/kza/src/training/personality.py`

| Class | Tests | Purpose |
|-------|-------|---------|
| TestPersonality | 3 | Personality dataclass |
| TestPersonalityManagerInit | 2 | Initialization & config loading |
| TestPersonalityManagerSetters | 7 | Configuration setters |
| TestPersonalityManagerSystemPrompt | 8 | System prompt generation |
| TestPersonalityManagerResponses | 5 | Response selection |
| TestPersonalityManagerConfig | 2 | Config retrieval |
| TestPersonalityManagerApplySetting | 9 | Interactive settings |
| TestPersonalityManagerPersistence | 2 | Save/load persistence |
| TestToneTemplates | 2 | Template validation |

**Key Mocks**: tempfile for config persistence

## Test Execution Commands

```bash
# All tests
pytest

# Specific module
pytest tests/unit/users/test_speaker_identifier.py
pytest tests/unit/memory/test_memory_manager.py
pytest tests/unit/training/test_personality.py

# Specific test class
pytest tests/unit/users/test_speaker_identifier.py::TestSpeakerIdentifierIdentify

# Specific test method
pytest tests/unit/users/test_speaker_identifier.py::TestSpeakerIdentifierIdentify::test_identify_known_speaker

# With coverage
pytest --cov=src --cov-report=html tests/

# Verbose output
pytest -v tests/

# Stop on first failure
pytest -x tests/

# Show print statements
pytest -s tests/
```

## Key Testing Patterns

### Pattern 1: Mocking GPU Operations
```python
@patch('src.users.speaker_identifier.SpeakerIdentifier.get_embedding')
def test_example(self, mock_get_embedding):
    mock_get_embedding.return_value = np.array([0.1, 0.2, 0.3])
    # Test code here
```

### Pattern 2: Temporary File I/O
```python
def test_example(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "config.json"
        # Test file operations
```

### Pattern 3: Mock Client/Service
```python
mock_client = MagicMock()
mock_client.get_or_create_collection.return_value = mock_collection
```

## Requirements

All tests use only:
- `pytest` (already in requirements)
- `unittest.mock` (Python stdlib)
- `tempfile` (Python stdlib)
- `numpy` (already in requirements)
- `json` (Python stdlib)

No GPU, SpeechBrain, ChromaDB, or other heavy dependencies required.

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Run from project root: `cd /sessions/great-epic-planck/mnt/kza && pytest` |
| Import paths wrong | conftest.py adds src/ to path automatically |
| Mock not working | Use `@patch('module.where.Class.is.used')` not where defined |
| Tests slow | Normal - they're comprehensive. Use `-x` to stop on first failure |
| Assertion fails | Check mock return values match expected types |

## Notes

- No tests actually execute yet (as requested)
- All 102 tests are ready to run
- Tests follow existing project style (see test_reasoner.py, test_logging.py)
- No side effects - tests clean up after themselves
- Tests are independent and can run in any order
