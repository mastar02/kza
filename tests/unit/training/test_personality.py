"""
Tests for Personality Manager module.
Tests personality configuration, system prompt generation, and response selection.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.training.personality import (
    PersonalityManager,
    Personality,
    TONE_TEMPLATES
)


class TestPersonality:
    """Test Personality dataclass"""

    def test_default_personality(self):
        """Test default personality configuration"""
        p = Personality()

        assert p.name == "Jarvis"
        assert p.description == "Asistente de hogar inteligente"
        assert p.tone == "friendly"
        assert p.language == "es"
        assert p.use_emojis is False
        assert p.verbose is False
        assert len(p.greeting_responses) > 0
        assert len(p.farewell_responses) > 0

    def test_custom_personality(self):
        """Test custom personality"""
        p = Personality(
            name="ARIA",
            tone="technical",
            language="en",
            verbose=True
        )

        assert p.name == "ARIA"
        assert p.tone == "technical"
        assert p.language == "en"
        assert p.verbose is True

    def test_custom_phrases(self):
        """Test custom personality phrases"""
        custom_greetings = ["Hi", "Hello", "Howdy"]
        p = Personality(greeting_responses=custom_greetings)

        assert p.greeting_responses == custom_greetings

    def test_household_info(self):
        """Test household information storage"""
        p = Personality()
        p.household_info = {"living_room_name": "Living"}

        assert p.household_info["living_room_name"] == "Living"


class TestPersonalityManagerInit:
    """Test PersonalityManager initialization"""

    def test_init_with_nonexistent_config(self):
        """Test initialization with non-existent config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))

            assert pm.config_path == config_path
            assert pm.personality.name == "Jarvis"
            assert pm.personality.tone == "friendly"

    def test_init_with_existing_config(self):
        """Test initialization loading existing config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            # Create config file
            config_data = {
                "name": "ARIA",
                "tone": "technical",
                "language": "en",
                "verbose": True
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            pm = PersonalityManager(str(config_path))

            assert pm.personality.name == "ARIA"
            assert pm.personality.tone == "technical"
            assert pm.personality.language == "en"
            assert pm.personality.verbose is True


class TestPersonalityManagerSetters:
    """Test PersonalityManager setter methods"""

    def test_set_name(self):
        """Test setting AI name"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.set_name("NOVA")

            assert pm.personality.name == "NOVA"

    def test_set_tone_valid(self):
        """Test setting valid tone"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.set_tone("technical")

            assert pm.personality.tone == "technical"

    def test_set_tone_invalid(self):
        """Test setting invalid tone"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))

            with pytest.raises(ValueError):
                pm.set_tone("invalid_tone")

    def test_add_custom_rule(self):
        """Test adding custom rule"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.add_custom_rule("Always be helpful")

            assert "Always be helpful" in pm.personality.custom_rules

    def test_remove_custom_rule(self):
        """Test removing custom rule"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.add_custom_rule("Rule 1")
            pm.add_custom_rule("Rule 2")

            removed = pm.remove_custom_rule(0)

            assert removed == "Rule 1"
            assert len(pm.personality.custom_rules) == 1

    def test_remove_custom_rule_invalid_index(self):
        """Test removing rule with invalid index"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))

            result = pm.remove_custom_rule(99)

            assert result is None

    def test_set_household_info(self):
        """Test setting household information"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.set_household_info("living_room", "Living")
            pm.set_household_info("bedrooms", "2")

            assert pm.personality.household_info["living_room"] == "Living"
            assert pm.personality.household_info["bedrooms"] == "2"


class TestPersonalityManagerSystemPrompt:
    """Test system prompt generation"""

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

    def test_build_system_prompt_technical(self):
        """Test building system prompt with technical tone"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.set_tone("technical")

            prompt = pm.build_system_prompt()

            assert "técnico" in prompt or "technical" in prompt.lower()

    def test_build_system_prompt_with_language(self):
        """Test system prompt includes language instruction"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.personality.language = "es"

            prompt = pm.build_system_prompt()

            assert "español" in prompt or "Spanish" in prompt

    def test_build_system_prompt_verbose(self):
        """Test system prompt reflects verbosity setting"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.personality.verbose = True

            prompt = pm.build_system_prompt()

            assert "detalladas" in prompt or "detailed" in prompt.lower()

    def test_build_system_prompt_with_emojis(self):
        """Test system prompt reflects emoji setting"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.personality.use_emojis = True

            prompt = pm.build_system_prompt()

            assert "emoji" in prompt.lower()

    def test_build_system_prompt_with_custom_rules(self):
        """Test system prompt includes custom rules"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.add_custom_rule("Be concise")
            pm.add_custom_rule("Use Spanish")

            prompt = pm.build_system_prompt()

            assert "Reglas especiales" in prompt
            assert "Be concise" in prompt
            assert "Use Spanish" in prompt

    def test_build_system_prompt_with_household_info(self):
        """Test system prompt includes household information"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.set_household_info("living_room", "Living Room")

            prompt = pm.build_system_prompt()

            assert "Información del hogar" in prompt
            assert "Living Room" in prompt

    def test_build_system_prompt_with_context(self):
        """Test system prompt includes context"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            context = {
                "time": "14:30",
                "user": "Juan"
            }

            prompt = pm.build_system_prompt(context)

            assert "14:30" in prompt
            assert "Juan" in prompt


class TestPersonalityManagerResponses:
    """Test response selection methods"""

    def test_get_greeting(self):
        """Test getting greeting response"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            greeting = pm.get_greeting()

            assert greeting in pm.personality.greeting_responses

    def test_get_farewell(self):
        """Test getting farewell response"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))

            # Access through get_random_response since get_farewell calls it
            pm.personality.farewell_responses = ["Goodbye"]
            farewell = pm.get_random_response("farewell")

            assert farewell == "Goodbye"

    def test_get_confirmation(self):
        """Test getting confirmation response"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            confirmation = pm.get_confirmation()

            assert confirmation in pm.personality.confirmation_phrases

    def test_get_error_message(self):
        """Test getting error message"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            error = pm.get_error_message()

            assert error in pm.personality.error_phrases

    def test_get_random_response_invalid_type(self):
        """Test getting random response with invalid type"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            response = pm.get_random_response("invalid_type")

            assert response == ""


class TestPersonalityManagerConfig:
    """Test configuration methods"""

    def test_get_config(self):
        """Test getting current configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.set_tone("technical")
            pm.add_custom_rule("Rule 1")
            pm.add_custom_rule("Rule 2")

            config = pm.get_config()

            assert config["name"] == "Jarvis"
            assert config["tone"] == "technical"
            assert config["custom_rules_count"] == 2
            assert "available_tones" in config

    def test_interactive_setup(self):
        """Test interactive setup returns questions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))

            setup = pm.interactive_setup()

            assert "questions" in setup
            assert len(setup["questions"]) > 0

            # Check question structure
            question = setup["questions"][0]
            assert "key" in question
            assert "prompt" in question
            assert "current" in question


class TestPersonalityManagerApplySetting:
    """Test apply_setting method"""

    def test_apply_setting_name(self):
        """Test applying name setting"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            response = pm.apply_setting("name", "NOVA")

            assert "NOVA" in response
            assert pm.personality.name == "NOVA"

    def test_apply_setting_tone(self):
        """Test applying tone setting"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            response = pm.apply_setting("tone", "technical")

            assert "technical" in response.lower()

    def test_apply_setting_tone_invalid(self):
        """Test applying invalid tone setting"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            response = pm.apply_setting("tone", "invalid")

            assert "no válido" in response or "valid" in response.lower()

    def test_apply_setting_verbose_true(self):
        """Test applying verbose setting as true"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.apply_setting("verbose", "detalladas")

            assert pm.personality.verbose is True

    def test_apply_setting_verbose_false(self):
        """Test applying verbose setting as false"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            pm.personality.verbose = True
            pm.apply_setting("verbose", "concisas")

            assert pm.personality.verbose is False

    def test_apply_setting_rule(self):
        """Test applying custom rule"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            response = pm.apply_setting("rule", "Be helpful")

            assert "Be helpful" in response
            assert "Be helpful" in pm.personality.custom_rules

    def test_apply_setting_household(self):
        """Test applying household info setting"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            response = pm.apply_setting("household", "living=Living Room")

            assert "living" in response
            assert pm.personality.household_info["living"] == "Living Room"

    def test_apply_setting_household_invalid_format(self):
        """Test applying household info with invalid format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            response = pm.apply_setting("household", "invalid_format")

            assert "Formato inválido" in response or "invalid" in response.lower()

    def test_apply_setting_unknown(self):
        """Test applying unknown setting"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm = PersonalityManager(str(config_path))
            response = pm.apply_setting("unknown_key", "value")

            assert "no reconocida" in response or "not recognized" in response.lower()


class TestPersonalityManagerPersistence:
    """Test configuration persistence"""

    def test_changes_persist_after_reload(self):
        """Test that changes persist after reload"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm1 = PersonalityManager(str(config_path))
            pm1.set_name("NOVA")
            pm1.set_tone("technical")

            # Reload in new instance
            pm2 = PersonalityManager(str(config_path))

            assert pm2.personality.name == "NOVA"
            assert pm2.personality.tone == "technical"

    def test_all_settings_persist(self):
        """Test all settings persist correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "personality.json"

            pm1 = PersonalityManager(str(config_path))
            pm1.set_name("ARIA")
            pm1.set_tone("butler")
            pm1.personality.language = "en"
            pm1.personality.verbose = True
            pm1.personality.use_emojis = True
            pm1.add_custom_rule("Rule 1")
            pm1.set_household_info("key", "value")

            pm2 = PersonalityManager(str(config_path))

            assert pm2.personality.name == "ARIA"
            assert pm2.personality.tone == "butler"
            assert pm2.personality.language == "en"
            assert pm2.personality.verbose is True
            assert pm2.personality.use_emojis is True
            assert "Rule 1" in pm2.personality.custom_rules
            assert pm2.personality.household_info["key"] == "value"


class TestToneTemplates:
    """Test tone templates"""

    def test_all_tones_have_required_fields(self):
        """Test all tone templates have required fields"""
        required_fields = ["system_prefix", "instruction_style", "example_response"]

        for tone_name, tone_template in TONE_TEMPLATES.items():
            for field in required_fields:
                assert field in tone_template, f"Tone '{tone_name}' missing field '{field}'"

    def test_tone_templates_have_placeholders(self):
        """Test tone templates support name placeholder"""
        for tone_name, tone_template in TONE_TEMPLATES.items():
            system_prefix = tone_template["system_prefix"]
            # Should be able to format with name
            formatted = system_prefix.format(name="TestName")
            assert "TestName" in formatted
