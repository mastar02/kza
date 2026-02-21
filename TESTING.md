# Testing Configuration and New Tests

## Overview

This document describes the pytest configuration and the new unit tests added to the KZA project.

## Pytest Configuration

### Configuration File
- **Location**: `pytest.ini`
- **Key Settings**:
  - `asyncio_mode = auto`: Automatically manages asyncio event loops for async tests
  - `testpaths = tests`: Pytest discovers tests in the `tests/` directory
  - `python_files = test_*.py`: Files matching `test_*.py` pattern
  - `python_classes = Test*`: Test classes start with `Test`
  - `python_functions = test_*`: Test methods start with `test_`

### Markers
Custom pytest markers defined:
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.gpu`: Tests requiring GPU

### Logging Configuration
- `log_level = DEBUG`: Default logging level
- `log_cli = false`: Console logging disabled by default
- `--tb=short`: Short traceback format for failures

## New Test Files

### 1. Speaker Identifier Tests
**File**: `tests/unit/users/test_speaker_identifier.py`
**Module**: `src/users/speaker_identifier.py`
**Test Count**: 23 tests

**Test Classes**:
- `TestSpeakerIdentifierInit`: Initialization with default and custom parameters
- `TestSpeakerIdentifierComputeSimilarity`: Cosine similarity computation
- `TestSpeakerIdentifierGetEmbedding`: Audio embedding extraction with mocks
- `TestSpeakerIdentifierIdentify`: Speaker identification (known/unknown)
- `TestSpeakerIdentifierVerify`: Speaker verification
- `TestSpeakerIdentifierCreateEnrollmentEmbedding`: Enrollment embedding from samples
- `TestSpeakerMatch`: SpeakerMatch dataclass validation
- `TestSpeakerIdentifierEdgeCases`: Edge cases and error handling

**Key Features**:
- All GPU-intensive operations are mocked using `@patch` and `MagicMock`
- Audio processing is tested without actual audio files
- Similarity calculations validated mathematically
- No dependencies on SpeechBrain or resemblyzer libraries

**Example Test**:
```python
@patch('src.users.speaker_identifier.SpeakerIdentifier.get_embedding')
def test_identify_known_speaker(self, mock_get_embedding):
    """Test identifying a known speaker"""
    identifier = SpeakerIdentifier(similarity_threshold=0.75)
    
    # Mock embeddings
    audio = np.zeros(16000, dtype=np.float32)
    current_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    registered_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    
    mock_get_embedding.return_value = current_embedding
    
    result = identifier.identify(audio, {"user_1": registered_embedding})
    
    assert result.user_id == "user_1"
    assert result.is_known is True
```

### 2. Memory Manager Tests
**File**: `tests/unit/memory/test_memory_manager.py`
**Module**: `src/memory/memory_manager.py`
**Test Count**: 38 tests

**Test Classes**:
- `TestConversationTurn`: Conversation turn creation and defaults
- `TestUserPreference`: User preference dataclass
- `TestMemoryFact`: Memory fact dataclass
- `TestShortTermMemory`: Short-term memory with deque-based storage
- `TestLongTermMemory`: Long-term memory with ChromaDB mocking
- `TestPreferencesStore`: JSON-based preferences persistence
- `TestMemoryManager`: Main memory manager orchestration
- `TestMemoryManagerIntegration`: Full conversation cycle testing

**Key Features**:
- ChromaDB client fully mocked for isolation
- Uses `tempfile.TemporaryDirectory` for file I/O testing
- Tests JSON persistence and loading
- Validates memory context building
- Integration tests for complete workflows

**Example Test**:
```python
def test_save_and_load_preferences(self):
    """Test saving and loading preferences"""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "preferences.json"
        
        store1 = PreferencesStore(str(file_path))
        store1.set("favorite_artist", "Queen", confidence=0.95, source="explicit")
        
        # Load in new instance
        store2 = PreferencesStore(str(file_path))
        value = store2.get("favorite_artist")
        
        assert value == "Queen"
```

### 3. Personality Manager Tests
**File**: `tests/unit/training/test_personality.py`
**Module**: `src/training/personality.py`
**Test Count**: 41 tests

**Test Classes**:
- `TestPersonality`: Personality dataclass with defaults
- `TestPersonalityManagerInit`: Initialization and config loading
- `TestPersonalityManagerSetters`: Name, tone, rules, household info setters
- `TestPersonalityManagerSystemPrompt`: System prompt generation
- `TestPersonalityManagerResponses`: Response type selection (greetings, errors, etc.)
- `TestPersonalityManagerConfig`: Configuration retrieval
- `TestPersonalityManagerApplySetting`: Interactive settings application
- `TestPersonalityManagerPersistence`: Configuration persistence across reloads
- `TestToneTemplates`: Tone template validation

**Key Features**:
- JSON-based configuration persistence tested
- All tone templates validated for required fields
- System prompt generation with context
- Interactive setup question generation
- Configuration reloading and persistence

**Example Test**:
```python
def test_build_system_prompt_friendly(self):
    """Test building system prompt with friendly tone"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "personality.json"
        
        pm = PersonalityManager(str(config_path))
        pm.set_name("Jarvis")
        pm.set_tone("friendly")
        
        prompt = pm.build_system_prompt()
        
        assert "Jarvis" in prompt
        assert "amigable" in prompt or "friendly" in prompt.lower()
```

## Running Tests

### All Tests
```bash
pytest
```

### Unit Tests Only
```bash
pytest tests/unit/
```

### Specific Module
```bash
pytest tests/unit/users/test_speaker_identifier.py
pytest tests/unit/memory/test_memory_manager.py
pytest tests/unit/training/test_personality.py
```

### With Verbose Output
```bash
pytest -v tests/
```

### With Coverage Report
```bash
pytest --cov=src tests/
```

### Specific Test Class
```bash
pytest tests/unit/users/test_speaker_identifier.py::TestSpeakerIdentifierIdentify
```

### Specific Test Method
```bash
pytest tests/unit/users/test_speaker_identifier.py::TestSpeakerIdentifierIdentify::test_identify_known_speaker
```

## Test Patterns Used

### 1. Mock Patterns
- `@patch()`: Decorator for method-level mocking
- `MagicMock()`: Flexible mock objects with call tracking
- `mock.return_value`: Setting mock return values
- `mock.side_effect`: Sequential return values

### 2. Fixture Patterns
- `tempfile.TemporaryDirectory`: Isolated file system for tests
- Fixtures from `tests/conftest.py` for shared mocks
- Test-local fixtures for test-specific setup

### 3. Assertion Patterns
- Standard pytest assertions with clear messages
- NumPy array comparisons with `np.array_equal()`
- Range checks for floating-point values
- Exception testing with `pytest.raises()`

## Dependencies

### No Additional Dependencies Required
All tests use only:
- `pytest`: Already in project requirements
- `unittest.mock`: Python standard library
- `tempfile`: Python standard library
- `numpy`: Already in project requirements
- `json`: Python standard library

### Mocked Dependencies
The following are mocked to avoid external dependencies:
- `SpeechBrain` models (speaker_identifier)
- `ChromaDB` client (memory_manager)
- Model loading and GPU operations

## Best Practices Followed

1. **Isolation**: Each test is independent and can run in any order
2. **Clarity**: Test names clearly describe what is being tested
3. **No Side Effects**: Tests use temporary directories and mocks
4. **Fast Execution**: No actual model loading, GPU operations, or external API calls
5. **Coverage**: Multiple scenarios including edge cases and error conditions
6. **Documentation**: Clear docstrings explaining test purpose
7. **Consistency**: Follows existing test style in the project

## Troubleshooting

### Import Errors
Ensure the project root is in PYTHONPATH:
```bash
export PYTHONPATH=/sessions/great-epic-planck/mnt/kza:$PYTHONPATH
```

### Module Not Found
The `conftest.py` adds `src/` to the path. If tests can't find modules:
```bash
cd /sessions/great-epic-planck/mnt/kza
pytest tests/
```

### Mock Not Working
Patch the import location where the object is used, not where it's defined:
```python
@patch('src.module.Class')  # Correct if Class is imported in src/module
@patch('src.other.Class')   # If Class is used in src/other but defined elsewhere
```

## Future Improvements

Potential enhancements:
1. Add parametrized tests for multiple scenarios
2. Add performance benchmarking tests
3. Add async test patterns for async methods
4. Increase coverage to >90% for critical modules
5. Add property-based testing with Hypothesis
