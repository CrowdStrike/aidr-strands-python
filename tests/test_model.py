"""Unit tests for AIGuardModel."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_aidr import AIGuardBlockedError, AIGuardModel
from strands_aidr.model import (
    _rebuild_events_from_guard_output,
    _reconstruct_assistant_message,
    _strands_messages_to_openai,
    _tool_specs_to_openai,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_downstream(events: list[dict]) -> MagicMock:
    """Return a mock downstream Model whose stream() yields *events*."""

    async def _stream(*args: Any, **kwargs: Any) -> AsyncGenerator[dict, None]:
        for e in events:
            yield e

    model = MagicMock()
    model.stream = _stream
    return model


def _text_events(text: str) -> list[dict]:
    """Minimal StreamEvent sequence for a plain-text assistant response."""
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": text}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                "metrics": {"latencyMs": 100},
            }
        },
    ]


def _make_guard_result(
    *,
    blocked: bool = False,
    transformed: bool = False,
    guard_output: dict | None = None,
    summary: str = "No issues detected.",
) -> MagicMock:
    result = MagicMock()
    result.blocked = blocked
    result.transformed = transformed
    result.guard_output = guard_output
    result.summary = summary
    return result


def _make_aidr_response(result: MagicMock) -> MagicMock:
    resp = MagicMock()
    resp.result = result
    return resp


SIMPLE_MESSAGES: list[dict] = [
    {"role": "user", "content": [{"text": "Hello, world!"}]}
]

# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestStrandsToOpenAI:
    def test_text_message(self) -> None:
        oai = _strands_messages_to_openai(SIMPLE_MESSAGES)  # type: ignore[arg-type]
        assert oai == [{"role": "user", "content": "Hello, world!"}]

    def test_system_prompt_prepended(self) -> None:
        oai = _strands_messages_to_openai(SIMPLE_MESSAGES, system_prompt="You are helpful.")  # type: ignore[arg-type]
        assert oai[0] == {"role": "system", "content": "You are helpful."}
        assert oai[1]["role"] == "user"

    def test_tool_use_block(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "id1", "name": "my_tool", "input": {"x": 1}}}
                ],
            }
        ]
        oai = _strands_messages_to_openai(msgs)  # type: ignore[arg-type]
        assert oai[0]["role"] == "assistant"
        assert oai[0]["tool_calls"][0]["function"]["name"] == "my_tool"

    def test_tool_result_block(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "id1", "status": "success", "content": [{"text": "42"}]}}
                ],
            }
        ]
        oai = _strands_messages_to_openai(msgs)  # type: ignore[arg-type]
        assert oai[0]["role"] == "tool"
        assert oai[0]["tool_call_id"] == "id1"
        assert oai[0]["content"] == "42"


class TestToolSpecsToOpenAI:
    def test_basic_conversion(self) -> None:
        specs = [
            {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string"}}}},
            }
        ]
        tools = _tool_specs_to_openai(specs)  # type: ignore[arg-type]
        assert tools[0]["function"]["name"] == "get_weather"
        assert tools[0]["function"]["parameters"]["type"] == "object"

    def test_raw_schema(self) -> None:
        specs = [
            {
                "name": "noop",
                "description": "Does nothing.",
                "inputSchema": {"type": "object"},
            }
        ]
        tools = _tool_specs_to_openai(specs)  # type: ignore[arg-type]
        assert tools[0]["function"]["parameters"]["type"] == "object"


class TestReconstructAssistantMessage:
    def test_text_response(self) -> None:
        events = _text_events("Hello there!")
        msg = _reconstruct_assistant_message(events)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello there!"

    def test_tool_call(self) -> None:
        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {"toolUse": {"name": "fn", "toolUseId": "call_1"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"q":'}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '"test"}'}}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
        msg = _reconstruct_assistant_message(events)
        assert msg["tool_calls"][0]["function"]["name"] == "fn"
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"q":"test"}'

    def test_empty_events(self) -> None:
        msg = _reconstruct_assistant_message([])
        assert msg["role"] == "assistant"
        assert "content" not in msg
        assert "tool_calls" not in msg


class TestRebuildEventsFromGuardOutput:
    def test_text_replaced(self) -> None:
        events = _text_events("original text")
        guard_output = {"messages": [{"role": "assistant", "content": "redacted text"}]}
        rebuilt = _rebuild_events_from_guard_output(events, guard_output)
        msg = _reconstruct_assistant_message(rebuilt)
        assert msg["content"] == "redacted text"

    def test_no_messages_in_guard_output(self) -> None:
        events = _text_events("original")
        rebuilt = _rebuild_events_from_guard_output(events, {})
        assert rebuilt == events


# ---------------------------------------------------------------------------
# AIGuardModel tests
# ---------------------------------------------------------------------------


@pytest.fixture
def downstream() -> MagicMock:
    return _make_downstream(_text_events("4"))


@pytest.fixture
def model(downstream: MagicMock) -> AIGuardModel:
    with patch("strands_aidr.model.AIGuard") as mock_aidr_cls:
        mock_aidr = mock_aidr_cls.return_value
        mock_aidr.guard_chat_completions.return_value = _make_aidr_response(
            _make_guard_result()
        )
        m = AIGuardModel(downstream, token="test-token")
    m._aidr = mock_aidr  # expose for per-test stubbing
    return m


class TestAIGuardModelPassthrough:
    @pytest.mark.asyncio
    async def test_passthrough_yields_all_events(self, model: AIGuardModel) -> None:
        model._aidr.guard_chat_completions.return_value = _make_aidr_response(_make_guard_result())  # type: ignore[attr-defined]
        events = [e async for e in model.stream(SIMPLE_MESSAGES)]  # type: ignore[arg-type]
        assert any("messageStop" in e for e in events)

    @pytest.mark.asyncio
    async def test_aidr_called_twice_for_input_and_output(self, model: AIGuardModel) -> None:
        model._aidr.guard_chat_completions.return_value = _make_aidr_response(_make_guard_result())  # type: ignore[attr-defined]
        _ = [e async for e in model.stream(SIMPLE_MESSAGES)]  # type: ignore[arg-type]
        assert model._aidr.guard_chat_completions.call_count == 2  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_event_types_passed_correctly(self, model: AIGuardModel) -> None:
        model._aidr.guard_chat_completions.return_value = _make_aidr_response(_make_guard_result())  # type: ignore[attr-defined]
        _ = [e async for e in model.stream(SIMPLE_MESSAGES)]  # type: ignore[arg-type]
        calls = model._aidr.guard_chat_completions.call_args_list  # type: ignore[attr-defined]
        event_types = [c.kwargs.get("event_type") for c in calls]
        assert "input" in event_types
        assert "output" in event_types


class TestAIGuardModelInputBlocked:
    @pytest.mark.asyncio
    async def test_raises_on_blocked_input(self, downstream: MagicMock) -> None:
        with patch("strands_aidr.model.AIGuard") as mock_aidr_cls:
            mock_aidr = mock_aidr_cls.return_value
            mock_aidr.guard_chat_completions.return_value = _make_aidr_response(
                _make_guard_result(blocked=True, summary="Malicious prompt detected.")
            )
            model = AIGuardModel(downstream, token="test-token")
        model._aidr = mock_aidr

        with pytest.raises(AIGuardBlockedError) as exc_info:
            _ = [e async for e in model.stream(SIMPLE_MESSAGES)]  # type: ignore[arg-type]

        assert exc_info.value.event_type == "input"
        assert "Malicious prompt" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_downstream_not_called_when_input_blocked(self, downstream: MagicMock) -> None:
        with patch("strands_aidr.model.AIGuard") as mock_aidr_cls:
            mock_aidr = mock_aidr_cls.return_value
            mock_aidr.guard_chat_completions.return_value = _make_aidr_response(
                _make_guard_result(blocked=True, summary="blocked")
            )
            model = AIGuardModel(downstream, token="test-token")
        model._aidr = mock_aidr

        downstream_called = False

        async def _tracking_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[dict, None]:
            nonlocal downstream_called
            downstream_called = True
            return
            yield  # make it a generator

        downstream.stream = _tracking_stream

        with pytest.raises(AIGuardBlockedError):
            _ = [e async for e in model.stream(SIMPLE_MESSAGES)]  # type: ignore[arg-type]

        assert not downstream_called


class TestAIGuardModelOutputBlocked:
    @pytest.mark.asyncio
    async def test_raises_on_blocked_output(self, downstream: MagicMock) -> None:
        call_count = 0

        def _side_effect(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if kwargs.get("event_type") == "input":
                return _make_aidr_response(_make_guard_result())
            return _make_aidr_response(_make_guard_result(blocked=True, summary="PII in response."))

        with patch("strands_aidr.model.AIGuard") as mock_aidr_cls:
            mock_aidr = mock_aidr_cls.return_value
            mock_aidr.guard_chat_completions.side_effect = _side_effect
            model = AIGuardModel(downstream, token="test-token")
        model._aidr = mock_aidr

        with pytest.raises(AIGuardBlockedError) as exc_info:
            _ = [e async for e in model.stream(SIMPLE_MESSAGES)]  # type: ignore[arg-type]

        assert exc_info.value.event_type == "output"


class TestAIGuardModelTransformed:
    @pytest.mark.asyncio
    async def test_input_transformed_messages_substituted(self, downstream: MagicMock) -> None:
        transformed_guard_output = {
            "messages": [{"role": "user", "content": "Hello, <REDACTED>!"}]
        }
        call_count = 0

        def _side_effect(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if kwargs.get("event_type") == "input":
                return _make_aidr_response(
                    _make_guard_result(transformed=True, guard_output=transformed_guard_output)
                )
            # Capture what messages were sent to guard on output call.
            sent_messages = kwargs["guard_input"]["messages"]
            _side_effect.output_messages = sent_messages  # type: ignore[attr-defined]
            return _make_aidr_response(_make_guard_result())

        with patch("strands_aidr.model.AIGuard") as mock_aidr_cls:
            mock_aidr = mock_aidr_cls.return_value
            mock_aidr.guard_chat_completions.side_effect = _side_effect
            model = AIGuardModel(downstream, token="test-token")
        model._aidr = mock_aidr

        _ = [e async for e in model.stream(SIMPLE_MESSAGES)]  # type: ignore[arg-type]
        # The output guard should receive the transformed (redacted) message.
        assert "<REDACTED>" in _side_effect.output_messages[0]["content"]  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_output_transformed_events_replaced(self, downstream: MagicMock) -> None:
        transformed_guard_output = {
            "messages": [{"role": "assistant", "content": "The answer is <REDACTED>."}]
        }

        def _side_effect(**kwargs: Any) -> MagicMock:
            if kwargs.get("event_type") == "input":
                return _make_aidr_response(_make_guard_result())
            return _make_aidr_response(
                _make_guard_result(transformed=True, guard_output=transformed_guard_output)
            )

        with patch("strands_aidr.model.AIGuard") as mock_aidr_cls:
            mock_aidr = mock_aidr_cls.return_value
            mock_aidr.guard_chat_completions.side_effect = _side_effect
            model = AIGuardModel(downstream, token="test-token")
        model._aidr = mock_aidr

        events = [e async for e in model.stream(SIMPLE_MESSAGES)]  # type: ignore[arg-type]
        msg = _reconstruct_assistant_message(events)
        assert "<REDACTED>" in msg["content"]


class TestAIGuardModelGuardFlags:
    @pytest.mark.asyncio
    async def test_guard_input_false_skips_input_guard(self, downstream: MagicMock) -> None:
        with patch("strands_aidr.model.AIGuard") as mock_aidr_cls:
            mock_aidr = mock_aidr_cls.return_value
            mock_aidr.guard_chat_completions.return_value = _make_aidr_response(_make_guard_result())
            model = AIGuardModel(downstream, token="test-token", guard_input=False)
        model._aidr = mock_aidr

        _ = [e async for e in model.stream(SIMPLE_MESSAGES)]  # type: ignore[arg-type]
        # Only the output guard should fire.
        calls = model._aidr.guard_chat_completions.call_args_list  # type: ignore[attr-defined]
        event_types = [c.kwargs.get("event_type") for c in calls]
        assert "input" not in event_types
        assert "output" in event_types

    @pytest.mark.asyncio
    async def test_guard_output_false_skips_output_guard(self, downstream: MagicMock) -> None:
        with patch("strands_aidr.model.AIGuard") as mock_aidr_cls:
            mock_aidr = mock_aidr_cls.return_value
            mock_aidr.guard_chat_completions.return_value = _make_aidr_response(_make_guard_result())
            model = AIGuardModel(downstream, token="test-token", guard_output=False)
        model._aidr = mock_aidr

        _ = [e async for e in model.stream(SIMPLE_MESSAGES)]  # type: ignore[arg-type]
        calls = model._aidr.guard_chat_completions.call_args_list  # type: ignore[attr-defined]
        event_types = [c.kwargs.get("event_type") for c in calls]
        assert "input" in event_types
        assert "output" not in event_types

    @pytest.mark.asyncio
    async def test_both_guards_false_no_aidr_calls(self, downstream: MagicMock) -> None:
        with patch("strands_aidr.model.AIGuard") as mock_aidr_cls:
            mock_aidr = mock_aidr_cls.return_value
            model = AIGuardModel(downstream, token="test-token", guard_input=False, guard_output=False)
        model._aidr = mock_aidr

        _ = [e async for e in model.stream(SIMPLE_MESSAGES)]  # type: ignore[arg-type]
        mock_aidr.guard_chat_completions.assert_not_called()


class TestAIGuardModelToolSpecs:
    @pytest.mark.asyncio
    async def test_tools_forwarded_to_aidr(self, downstream: MagicMock) -> None:
        tool_specs = [
            {
                "name": "search",
                "description": "Search the web.",
                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]
        with patch("strands_aidr.model.AIGuard") as mock_aidr_cls:
            mock_aidr = mock_aidr_cls.return_value
            mock_aidr.guard_chat_completions.return_value = _make_aidr_response(_make_guard_result())
            model = AIGuardModel(downstream, token="test-token")
        model._aidr = mock_aidr

        _ = [e async for e in model.stream(SIMPLE_MESSAGES, tool_specs=tool_specs)]  # type: ignore[arg-type]
        call_kwargs = model._aidr.guard_chat_completions.call_args_list[0].kwargs  # type: ignore[attr-defined]
        assert "tools" in call_kwargs["guard_input"]
        assert call_kwargs["guard_input"]["tools"][0]["function"]["name"] == "search"


class TestAIGuardModelConfig:
    def test_missing_token_raises(self, downstream: MagicMock) -> None:
        import os

        with pytest.raises(ValueError, match="AIDR token"):
            # Ensure env var is not set.
            env = {k: v for k, v in os.environ.items() if k != "CS_AIDR_TOKEN"}
            with patch.dict(os.environ, env, clear=True), patch("strands_aidr.model.AIGuard"):
                    AIGuardModel(downstream)

    def test_get_config_returns_dict(self, model: AIGuardModel) -> None:
        cfg = model.get_config()
        assert isinstance(cfg, dict)

    def test_update_config_merges(self, model: AIGuardModel) -> None:
        model.update_config(app_id="my-app")
        assert model.get_config()["app_id"] == "my-app"
