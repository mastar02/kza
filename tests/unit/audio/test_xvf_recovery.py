"""Tests: budget of the automatic XVF3800 firmware reboot (T048-U1)."""
import pytest

from src.audio.xvf_recovery import (
    REBOOT_WINDOW_S,
    XvfRebootBudget,
    XvfRebootDecision,
    XvfRebootPolicy,
)


def _budget(clock, **policy):
    options = {"enabled": True, "failures_before_reboot": 2, "max_per_hour": 3}
    options.update(policy)
    return XvfRebootBudget(XvfRebootPolicy(**options), clock=lambda: clock[0])


def _fail(budget, room, times, start=0):
    for opening in range(start, start + times):
        budget.note_recovery(room, float(opening + 1))


class TestThreshold:
    def test_reboot_only_after_the_configured_failed_recoveries(self):
        budget = _budget([0.0])
        assert budget.decide("living") is XvfRebootDecision.RECOVER
        _fail(budget, "living", 1)
        assert budget.decide("living") is XvfRebootDecision.RECOVER
        _fail(budget, "living", 1, start=1)
        assert budget.failures("living") == 2
        assert budget.decide("living") is XvfRebootDecision.REBOOT

    def test_one_opening_counts_once(self):
        budget = _budget([0.0])
        for _ in range(5):
            budget.note_recovery("living", 42.0)
        assert budget.failures("living") == 1

    def test_audio_ends_the_streak(self):
        budget = _budget([0.0])
        _fail(budget, "living", 2)
        budget.note_audio("living")
        assert budget.failures("living") == 0
        assert budget.decide("living") is XvfRebootDecision.RECOVER

    def test_disabled_policy_never_reboots(self):
        budget = _budget([0.0], enabled=False)
        _fail(budget, "living", 50)
        assert budget.decide("living") is XvfRebootDecision.RECOVER


class TestPerRoomIsolation:
    def test_rooms_keep_separate_streaks_and_caps(self):
        budget = _budget([0.0], max_per_hour=1)
        _fail(budget, "cocina", 2)
        assert budget.decide("cocina") is XvfRebootDecision.REBOOT
        assert budget.decide("hall") is XvfRebootDecision.RECOVER
        budget.charge("cocina")
        _fail(budget, "hall", 2)
        assert budget.decide("hall") is XvfRebootDecision.REBOOT
        assert budget.reboots_in_window("hall") == 0
        budget.note_audio("hall")
        assert budget.reboots_in_window("cocina") == 1


class TestCapAndBackoff:
    def test_each_reboot_in_the_hour_doubles_the_required_failures(self):
        budget = _budget([0.0])
        assert budget.required_failures("living") == 2
        assert budget.charge("living") == 1
        assert budget.failures("living") == 0
        assert budget.required_failures("living") == 4
        _fail(budget, "living", 3)
        assert budget.decide("living") is XvfRebootDecision.RECOVER
        _fail(budget, "living", 1, start=3)
        assert budget.decide("living") is XvfRebootDecision.REBOOT
        assert budget.charge("living") == 2
        assert budget.required_failures("living") == 8

    def test_cap_stops_reboots_until_the_window_frees_one(self):
        clock = [0.0]
        budget = _budget(clock, max_per_hour=2)
        budget.charge("living")
        clock[0] = 600.0
        budget.charge("living")
        _fail(budget, "living", 8)
        assert budget.decide("living") is XvfRebootDecision.CAPPED
        assert budget.report_cap("living") is True
        assert budget.report_cap("living") is False  # one error per episode
        clock[0] = REBOOT_WINDOW_S + 1.0  # the first reboot left the window
        assert budget.reboots_in_window("living") == 1
        assert budget.decide("living") is XvfRebootDecision.REBOOT
        assert budget.report_cap("living") is True  # a new episode may report again

    def test_audio_starts_a_new_cap_episode(self):
        budget = _budget([0.0], max_per_hour=1)
        budget.charge("living")
        _fail(budget, "living", 4)
        assert budget.decide("living") is XvfRebootDecision.CAPPED
        assert budget.report_cap("living") is True
        budget.note_audio("living")
        _fail(budget, "living", 4, start=10)
        assert budget.decide("living") is XvfRebootDecision.CAPPED
        assert budget.report_cap("living") is True


class TestPolicyValidation:
    @pytest.mark.parametrize("options", [
        {"failures_before_reboot": 0}, {"failures_before_reboot": 1.5},
        {"failures_before_reboot": True}, {"max_per_hour": 0}, {"max_per_hour": "3"},
        {"settle_s": 0}, {"settle_s": float("nan")}, {"settle_s": float("inf")},
        {"settle_s": "10"}, {"enabled": "yes"},
    ])
    def test_invalid_values_are_refused(self, options):
        with pytest.raises(ValueError):
            XvfRebootPolicy(**options)

    def test_default_is_disabled(self):
        assert XvfRebootPolicy().enabled is False
