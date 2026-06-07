"""Integration smoke test for AIGuardModel.

Requires real credentials and a downstream model.  Skipped automatically when
``CS_AIDR_TOKEN`` or ``CS_AIDR_BASE_URL`` are not set.

Run manually::

    CS_AIDR_TOKEN=pts_... CS_AIDR_BASE_URL=https://api.crowdstrike.com/aidr/{SERVICE_NAME} \\
        pytest tests/test_integration.py -v
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_aidr import AIGuardModel

# ---------------------------------------------------------------------------
# Skip condition
# ---------------------------------------------------------------------------

_CREDS_PRESENT = bool(os.environ.get("CS_AIDR_TOKEN")) and bool(os.environ.get("CS_AIDR_BASE_URL"))

pytestmark = pytest.mark.skipif(
    not _CREDS_PRESENT,
    reason="CS_AIDR_TOKEN and CS_AIDR_BASE_URL environment variables are required for integration tests.",
)

# ---------------------------------------------------------------------------
# Minimal downstream stub that returns a fixed text response
# ---------------------------------------------------------------------------

_FIXED_EVENTS = [
    {"messageStart": {"role": "assistant"}},
    {"contentBlockStart": {"start": {}}},
    {"contentBlockDelta": {"delta": {"text": "2 + 2 = 4"}}},
    {"contentBlockStop": {}},
    {"messageStop": {"stopReason": "end_turn"}},
]


def _make_fixed_downstream() -> MagicMock:
    async def _stream(*args: Any, **kwargs: Any) -> AsyncGenerator[dict, None]:
        for e in _fixed_events():
            yield e

    def _fixed_events():  # type: ignore[no-untyped-def]
        return _FIXED_EVENTS

    model = MagicMock()
    model.stream = _stream
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIntegrationPassthrough:
    @pytest.mark.asyncio
    async def test_passthrough_with_real_aidr(self) -> None:
        """Sends a benign message through AIDR and expects it to pass."""
        downstream = _make_fixed_downstream()
        model = AIGuardModel(downstream, guard_input=True, guard_output=True)

        messages = [{"role": "user", "content": [{"text": "What is 2 + 2?"}]}]
        events = [e async for e in model.stream(messages)]  # type: ignore[arg-type]

        assert any("messageStop" in e for e in events), "Expected messageStop in events"
        print(f"\nReceived {len(events)} events from AIGuardModel (real AIDR)")
