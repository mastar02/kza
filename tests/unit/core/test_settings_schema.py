"""Tests del schema Pydantic de config/settings.yaml.

Valida que el settings.yaml del repo pasa el schema (smoke de boot) y que
el schema rechaza configs sin las secciones/campos que src/main.py exige.
"""

from pathlib import Path

import pytest
import yaml

from src.core.settings_schema import check_unresolved_env_vars, validate_settings

SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


@pytest.fixture()
def real_config() -> dict:
    with open(SETTINGS_PATH, "r") as f:
        return yaml.safe_load(f)


class TestValidateSettings:
    @pytest.mark.parametrize("consent", ["false", "true", 0, 1])
    def test_cloud_consent_must_be_a_literal_boolean(self, real_config, consent):
        real_config["reasoner"]["cloud"]["consent"] = consent
        with pytest.raises(ValueError, match="consent"):
            validate_settings(real_config)

    def test_destination_must_describe_processing_not_transport(self, real_config):
        real_config["reasoner"]["destination"] = "loopback"
        with pytest.raises(ValueError, match="destination"):
            validate_settings(real_config)

    def test_real_settings_yaml_validates(self, real_config):
        """El settings.yaml versionado en el repo debe pasar el schema."""
        result = validate_settings(real_config)
        assert result is real_config  # pass-through: no altera el dict

    def test_missing_required_section_fails(self, real_config):
        del real_config["embeddings"]
        with pytest.raises(ValueError, match="embeddings"):
            validate_settings(real_config)

    def test_missing_required_field_fails(self, real_config):
        del real_config["speaker_id"]["model"]
        with pytest.raises(ValueError, match="speaker_id"):
            validate_settings(real_config)

    def test_wrong_type_fails(self, real_config):
        real_config["home_assistant"]["url"] = 12345
        with pytest.raises(ValueError, match="home_assistant"):
            validate_settings(real_config)

    def test_control_selection_path_cannot_be_a_boolean(self, real_config):
        real_config["home_assistant"]["control_selection_path"] = True
        with pytest.raises(ValueError, match="control_selection_path"):
            validate_settings(real_config)

    @pytest.mark.parametrize("timeouts", [{"unlock": 60}, {"open_cover": True}, {"open_cover": "60"}, {"open_cover": 0}, {"open_cover": 121}, {"open_cover": float("nan")}])
    def test_confirmation_timeouts_are_known_bounded_numeric_values(self, real_config, timeouts):
        real_config["home_assistant"]["operation_confirmation_timeouts"] = timeouts
        with pytest.raises(ValueError, match="operation_confirmation_timeouts"):
            validate_settings(real_config)

    def test_extra_sections_and_fields_allowed(self, real_config):
        """El schema es permisivo: claves nuevas no rompen el boot."""
        real_config["seccion_futura"] = {"clave": 1}
        real_config["embeddings"]["campo_nuevo"] = True
        validate_settings(real_config)

    def test_non_dict_config_fails(self):
        with pytest.raises(ValueError, match="mapping"):
            validate_settings(None)

    def test_multiple_errors_reported_together(self, real_config):
        """El mensaje lista TODOS los campos inválidos, no solo el primero."""
        del real_config["embeddings"]
        real_config["home_assistant"]["url"] = 12345
        with pytest.raises(ValueError) as exc_info:
            validate_settings(real_config)
        assert "embeddings" in str(exc_info.value)
        assert "home_assistant.url" in str(exc_info.value)

    def test_null_section_fails(self, real_config):
        """Sección declarada pero vacía (`stt:` sin valor) = config rota."""
        real_config["stt"] = None
        with pytest.raises(ValueError, match="stt"):
            validate_settings(real_config)

    def test_disabled_section_does_not_require_model(self, real_config):
        """speaker_id/emotion con enabled=false no exigen model/device (como main.py)."""
        real_config["speaker_id"] = {"enabled": False}
        real_config["emotion"] = {"enabled": False}
        validate_settings(real_config)

    def test_enabled_section_requires_model_and_device(self, real_config):
        real_config["emotion"] = {"enabled": True}
        with pytest.raises(ValueError, match="emotion"):
            validate_settings(real_config)


class TestCheckUnresolvedEnvVars:
    def test_unresolved_ha_var_raises(self):
        config = {"home_assistant": {"url": "${HOME_ASSISTANT_URL}"}}
        with pytest.raises(ValueError, match="HOME_ASSISTANT_URL"):
            check_unresolved_env_vars(config)

    def test_unresolved_non_critical_var_only_warns(self, caplog):
        config = {"spotify": {"client_id": "${SPOTIFY_CLIENT_ID}"}}
        with caplog.at_level("WARNING"):
            check_unresolved_env_vars(config)
        assert "SPOTIFY_CLIENT_ID" in caplog.text

    def test_resolved_config_is_silent(self, caplog):
        config = {"home_assistant": {"url": "http://192.168.1.2:8123"}}
        with caplog.at_level("WARNING"):
            check_unresolved_env_vars(config)
        assert caplog.text == ""
