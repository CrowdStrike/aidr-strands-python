"""Exceptions raised by the AIDR Strands integration."""

from __future__ import annotations


class AIGuardBlockedError(RuntimeError):
    """Raised when CrowdStrike AIDR blocks a request due to a policy violation.

    Attributes:
        event_type: ``"input"`` if the block occurred before sending to the downstream
            model, or ``"output"`` if it occurred after the model responded.
        summary: Human-readable summary string returned by AIDR describing which
            detectors triggered and what action was taken.
    """

    def __init__(self, event_type: str, summary: str) -> None:
        super().__init__(f"AIDR blocked {event_type}: {summary}")
        self.event_type = event_type
        self.summary = summary


class AIGuardUnavailableError(RuntimeError):
    """Raised when AIDR is unreachable or returns no result and fail_open is False.

    Attributes:
        event_type: ``"input"`` or ``"output"`` indicating which guard failed.
    """

    def __init__(self, event_type: str) -> None:
        super().__init__(
            f"AIDR {event_type} guard unavailable and fail_open is disabled"
        )
        self.event_type = event_type
