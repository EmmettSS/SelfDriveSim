"""Unit tests for the control payload validator."""

import pytest

from simulation.consumers import CommandValidationError, sanitize_command


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            {"steering": 5.0, "throttle": 2, "brake": -1, "turn_signal": 7},
            {"steering": 1.0, "throttle": 1.0, "brake": 0.0, "turn_signal": 2},
        ),
        (
            {"steering": -3, "throttle": 0.3, "brake": 0.9, "turn_signal": 1.4},
            {"steering": -1.0, "throttle": 0.3, "brake": 0.9, "turn_signal": 1},
        ),
        (
            {"steering": 0.5, "throttle": 0.5, "brake": 0.5, "turn_signal": 1.6},
            {"steering": 0.5, "throttle": 0.5, "brake": 0.5, "turn_signal": 2},
        ),
        (
            {"steering": 0, "throttle": 0, "brake": 0, "turn_signal": 0},
            {"steering": 0.0, "throttle": 0.0, "brake": 0.0, "turn_signal": 0},
        ),
    ],
)
def test_values_are_clipped_into_valid_ranges(raw, expected):
    assert sanitize_command(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "not an object",
        None,
        [0.0, 0.0, 0.0, 0],
        {"steering": 0.0, "throttle": 0.0, "brake": 0.0},  # turn_signal missing
        {"steering": True, "throttle": 0.0, "brake": 0.0, "turn_signal": 0},
        {"steering": "0.5", "throttle": 0.0, "brake": 0.0, "turn_signal": 0},
        {"steering": float("nan"), "throttle": 0.0, "brake": 0.0, "turn_signal": 0},
        {"steering": float("inf"), "throttle": 0.0, "brake": 0.0, "turn_signal": 0},
    ],
)
def test_invalid_payloads_are_rejected(raw):
    with pytest.raises(CommandValidationError):
        sanitize_command(raw)
