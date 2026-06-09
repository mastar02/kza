"""Tests del schema Pydantic de config/settings.yaml.

Valida que el settings.yaml del repo pasa el schema (smoke de boot) y que
el schema rechaza configs sin las secciones/campos que src/main.py exige.
"""

from pathlib import Path

import pytest
import yaml

from src.core.settings_schema import validate_settings

SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


@pytest.fixture()
def real_config() -> dict:
    with open(SETTINGS_PATH, "r") as f:
        return yaml.safe_load(f)


class TestValidateSettings:
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

    def test_extra_sections_and_fields_allowed(self, real_config):
        """El schema es permisivo: claves nuevas no rompen el boot."""
        real_config["seccion_futura"] = {"clave": 1}
        real_config["embeddings"]["campo_nuevo"] = True
        validate_settings(real_config)

    def test_non_dict_config_fails(self):
        with pytest.raises(ValueError):
            validate_settings(None)
