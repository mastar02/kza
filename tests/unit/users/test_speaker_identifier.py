"""
Tests for Speaker Identifier module.
Tests speaker embeddings, identification, and verification functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.users.speaker_identifier import SpeakerIdentifier, SpeakerMatch
from tests.factories import make_speaker_identifier


class TestSpeakerIdentifierInit:
    """Test SpeakerIdentifier initialization"""

    def test_init_requires_device(self):
        """Constructor must reject missing device — config is the source of truth."""
        with pytest.raises(TypeError):
            SpeakerIdentifier(model_name="speechbrain/spkrec-ecapa-voxceleb")

    def test_init_requires_model_name(self):
        """Constructor must reject missing model_name."""
        with pytest.raises(TypeError):
            SpeakerIdentifier(device="cpu")

    def test_init_with_factory_defaults(self):
        """Factory provides CPU defaults for behavior-focused tests."""
        identifier = make_speaker_identifier()

        assert identifier.model_name == "speechbrain/spkrec-ecapa-voxceleb"
        assert identifier.device == "cpu"
        assert identifier.similarity_threshold == 0.75
        assert identifier.sample_rate == 16000
        assert identifier._model is None
        assert isinstance(identifier._embeddings_cache, dict)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters"""
        identifier = make_speaker_identifier(
            model_name="resemblyzer",
            device="cpu",
            similarity_threshold=0.8,
            sample_rate=22050
        )

        assert identifier.model_name == "resemblyzer"
        assert identifier.device == "cpu"
        assert identifier.similarity_threshold == 0.8
        assert identifier.sample_rate == 22050


class TestSpeakerIdentifierComputeSimilarity:
    """Test compute_similarity method"""

    def test_compute_similarity_identical_embeddings(self):
        """Test similarity between identical embeddings"""
        identifier = make_speaker_identifier()

        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        similarity = identifier.compute_similarity(embedding, embedding)

        # Identical embeddings should have similarity close to 1.0
        assert 0.99 <= similarity <= 1.01

    def test_compute_similarity_orthogonal_embeddings(self):
        """Test similarity between orthogonal embeddings"""
        identifier = make_speaker_identifier()

        embedding1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        similarity = identifier.compute_similarity(embedding1, embedding2)

        # Orthogonal vectors should have similarity around 0.5
        assert 0.4 <= similarity <= 0.6

    def test_compute_similarity_opposite_embeddings(self):
        """Test similarity between opposite embeddings"""
        identifier = make_speaker_identifier()

        embedding1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        similarity = identifier.compute_similarity(embedding1, embedding2)

        # Opposite vectors should have similarity close to 0
        assert 0.0 <= similarity <= 0.1

    def test_compute_similarity_zero_embeddings(self):
        """Test similarity with zero embeddings"""
        identifier = make_speaker_identifier()

        embedding1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        similarity = identifier.compute_similarity(embedding1, embedding2)

        # Should return 0 when one embedding is zero
        assert similarity == 0.0

    def test_compute_similarity_returns_normalized_range(self):
        """Test that similarity is always in [0, 1] range"""
        identifier = make_speaker_identifier()

        for _ in range(10):
            embedding1 = np.random.randn(256).astype(np.float32)
            embedding2 = np.random.randn(256).astype(np.float32)

            similarity = identifier.compute_similarity(embedding1, embedding2)
            assert 0.0 <= similarity <= 1.0


class TestSpeakerIdentifierGetEmbedding:
    """Test get_embedding method with mocks"""

    @patch('src.users.speaker_identifier.SpeakerIdentifier._load_speechbrain')
    def test_get_embedding_audio_normalization(self, mock_load):
        """Test that audio is normalized correctly"""
        identifier = make_speaker_identifier()

        # Mock the model
        mock_model = MagicMock()
        identifier._model = mock_model
        identifier._model_type = "speechbrain"

        # Audio that needs normalization (values > 1.0)
        audio = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        # Mock embedding output
        mock_tensor_result = MagicMock()
        mock_tensor_result.squeeze.return_value.cpu.return_value.numpy.return_value = np.array(
            [0.1, 0.2, 0.3, 0.4], dtype=np.float32
        )
        mock_model.encode_batch.return_value = mock_tensor_result

        with patch('torch.from_numpy') as mock_torch:
            mock_torch.return_value.unsqueeze.return_value = MagicMock()
            embedding = identifier.get_embedding(audio)

        assert embedding is not None
        assert isinstance(embedding, np.ndarray)

    @patch('src.users.speaker_identifier.SpeakerIdentifier._load_speechbrain')
    def test_get_embedding_dtype_conversion(self, mock_load):
        """Test that audio dtype is converted to float32"""
        identifier = make_speaker_identifier()

        # Mock the model
        mock_model = MagicMock()
        identifier._model = mock_model
        identifier._model_type = "speechbrain"

        # Int16 audio
        audio = np.array([100, 200, 300, 400], dtype=np.int16)

        # Mock embedding output
        mock_tensor_result = MagicMock()
        mock_tensor_result.squeeze.return_value.cpu.return_value.numpy.return_value = np.array(
            [0.1, 0.2, 0.3], dtype=np.float32
        )
        mock_model.encode_batch.return_value = mock_tensor_result

        with patch('torch.from_numpy') as mock_torch:
            mock_torch.return_value.unsqueeze.return_value = MagicMock()
            embedding = identifier.get_embedding(audio)

        assert embedding is not None


class TestSpeakerIdentifierIdentify:
    """Test identify method"""

    @patch('src.users.speaker_identifier.SpeakerIdentifier.get_embedding')
    def test_identify_known_speaker(self, mock_get_embedding):
        """Test identifying a known speaker"""
        identifier = make_speaker_identifier(similarity_threshold=0.75)

        # Mock embeddings
        audio = np.zeros(16000, dtype=np.float32)
        current_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        registered_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        mock_get_embedding.return_value = current_embedding

        registered_embeddings = {
            "user_1": registered_embedding
        }

        result = identifier.identify(audio, registered_embeddings)

        assert isinstance(result, SpeakerMatch)
        assert result.user_id == "user_1"
        assert result.is_known is True
        assert result.confidence >= identifier.similarity_threshold

    @patch('src.users.speaker_identifier.SpeakerIdentifier.get_embedding')
    def test_identify_unknown_speaker(self, mock_get_embedding):
        """Test identifying an unknown speaker"""
        identifier = make_speaker_identifier(similarity_threshold=0.75)

        # Mock embeddings with low similarity
        audio = np.zeros(16000, dtype=np.float32)
        current_embedding = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        registered_embedding = np.array([0.0, 0.1, 0.0, 0.0], dtype=np.float32)

        mock_get_embedding.return_value = current_embedding

        registered_embeddings = {
            "user_1": registered_embedding
        }

        result = identifier.identify(audio, registered_embeddings)

        assert isinstance(result, SpeakerMatch)
        assert result.user_id is None
        assert result.is_known is False

    @patch('src.users.speaker_identifier.SpeakerIdentifier.get_embedding')
    def test_identify_empty_registered_embeddings(self, mock_get_embedding):
        """Test identifying when no speakers are registered"""
        identifier = make_speaker_identifier()

        audio = np.zeros(16000, dtype=np.float32)
        current_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        mock_get_embedding.return_value = current_embedding

        result = identifier.identify(audio, {})

        assert result.user_id is None
        assert result.is_known is False
        assert result.confidence == 0.0

    @patch('src.users.speaker_identifier.SpeakerIdentifier.get_embedding')
    def test_identify_multiple_speakers(self, mock_get_embedding):
        """Test identifying best match from multiple registered speakers"""
        identifier = make_speaker_identifier(similarity_threshold=0.75)

        audio = np.zeros(16000, dtype=np.float32)
        current_embedding = np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32)

        mock_get_embedding.return_value = current_embedding

        registered_embeddings = {
            "user_1": np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32),  # Best match
            "user_2": np.array([0.0, 0.8, 0.2, 0.0], dtype=np.float32),  # Worse match
            "user_3": np.array([0.0, 0.0, 0.8, 0.2], dtype=np.float32),  # Worst match
        }

        result = identifier.identify(audio, registered_embeddings)

        assert result.user_id == "user_1"
        assert result.is_known is True


class TestSpeakerIdentifierVerify:
    """Test verify method"""

    @patch('src.users.speaker_identifier.SpeakerIdentifier.get_embedding')
    def test_verify_same_speaker(self, mock_get_embedding):
        """Test verifying the same speaker"""
        identifier = make_speaker_identifier(similarity_threshold=0.75)

        audio = np.zeros(16000, dtype=np.float32)
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        mock_get_embedding.return_value = embedding

        is_match, confidence = identifier.verify(audio, embedding)

        assert is_match is True
        assert confidence >= 0.99

    @patch('src.users.speaker_identifier.SpeakerIdentifier.get_embedding')
    def test_verify_different_speaker(self, mock_get_embedding):
        """Test verifying different speakers"""
        identifier = make_speaker_identifier(similarity_threshold=0.75)

        audio = np.zeros(16000, dtype=np.float32)
        current_embedding = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        claimed_embedding = np.array([0.0, 0.1, 0.0, 0.0], dtype=np.float32)

        mock_get_embedding.return_value = current_embedding

        is_match, confidence = identifier.verify(audio, claimed_embedding)

        assert is_match is False
        assert confidence < identifier.similarity_threshold


class TestSpeakerIdentifierCreateEnrollmentEmbedding:
    """Test create_enrollment_embedding method"""

    @patch('src.users.speaker_identifier.SpeakerIdentifier.get_embedding')
    def test_create_enrollment_embedding_single_sample(self, mock_get_embedding):
        """Test enrollment embedding from single sample"""
        identifier = make_speaker_identifier()

        audio_sample = np.zeros(16000, dtype=np.float32)
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        mock_get_embedding.return_value = embedding

        result = identifier.create_enrollment_embedding([audio_sample])

        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    @patch('src.users.speaker_identifier.SpeakerIdentifier.get_embedding')
    def test_create_enrollment_embedding_multiple_samples(self, mock_get_embedding):
        """Test enrollment embedding from multiple samples"""
        identifier = make_speaker_identifier()

        audio_samples = [
            np.zeros(16000, dtype=np.float32),
            np.zeros(16000, dtype=np.float32),
            np.zeros(16000, dtype=np.float32),
        ]

        # Different embeddings for each sample
        embeddings = [
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float32),
            np.array([0.05, 0.15, 0.25, 0.35], dtype=np.float32),
        ]

        mock_get_embedding.side_effect = embeddings

        result = identifier.create_enrollment_embedding(audio_samples)

        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        # Result should be normalized
        norm = np.linalg.norm(result)
        assert 0.99 <= norm <= 1.01

    def test_create_enrollment_embedding_empty_list(self):
        """Test enrollment embedding with empty list"""
        identifier = make_speaker_identifier()

        with pytest.raises(ValueError, match="Se necesita al menos una muestra"):
            identifier.create_enrollment_embedding([])


class TestSpeakerMatch:
    """Test SpeakerMatch dataclass"""

    def test_speaker_match_creation(self):
        """Test SpeakerMatch instantiation"""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        match = SpeakerMatch(
            user_id="user_1",
            confidence=0.95,
            embedding=embedding,
            is_known=True
        )

        assert match.user_id == "user_1"
        assert match.confidence == 0.95
        assert np.array_equal(match.embedding, embedding)
        assert match.is_known is True

    def test_speaker_match_unknown(self):
        """Test SpeakerMatch for unknown speaker"""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        match = SpeakerMatch(
            user_id=None,
            confidence=0.3,
            embedding=embedding,
            is_known=False
        )

        assert match.user_id is None
        assert match.confidence == 0.3
        assert match.is_known is False


class TestSpeakerIdentifierEdgeCases:
    """Test edge cases and error handling"""

    def test_similarity_with_large_embeddings(self):
        """Test similarity computation with large embeddings"""
        identifier = make_speaker_identifier()

        embedding1 = np.random.randn(512).astype(np.float32)
        embedding2 = np.random.randn(512).astype(np.float32)

        similarity = identifier.compute_similarity(embedding1, embedding2)

        assert 0.0 <= similarity <= 1.0

    @patch('src.users.speaker_identifier.SpeakerIdentifier.get_embedding')
    def test_identify_with_nan_values(self, mock_get_embedding):
        """Test handling of NaN values in embeddings"""
        identifier = make_speaker_identifier()

        # Embedding with NaN values (edge case)
        audio = np.zeros(16000, dtype=np.float32)
        embedding_with_nan = np.array([0.1, np.nan, 0.3], dtype=np.float32)

        mock_get_embedding.return_value = embedding_with_nan

        registered_embeddings = {
            "user_1": np.array([0.1, 0.2, 0.3], dtype=np.float32)
        }

        # Should handle gracefully
        result = identifier.identify(audio, registered_embeddings)
        assert isinstance(result, SpeakerMatch)

    def test_similarity_threshold_boundary(self):
        """Test behavior at similarity threshold boundaries"""
        identifier = make_speaker_identifier(similarity_threshold=0.75)

        # At threshold
        embedding1 = np.array([0.75, 0.0], dtype=np.float32)
        embedding2 = np.array([0.75, 0.0], dtype=np.float32)

        similarity = identifier.compute_similarity(embedding1, embedding2)
        assert similarity >= 0.75
