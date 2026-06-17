"""AIGuardModel — wraps any Strands model with CrowdStrike AIDR security guardrails.

The wrapper intercepts the model's ``stream()`` call on both sides:

- **Input guard** (``event_type="input"``): runs before the downstream model is called.
  Blocked requests raise :class:`AIGuardBlockedError`; transformed content is substituted.
- **Output guard** (``event_type="output"``): runs after the downstream model responds.
  Blocked responses raise :class:`AIGuardBlockedError`; transformed content is substituted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, TypedDict, TypeVar, Unpack, override

from crowdstrike_aidr import AIGuard
from crowdstrike_aidr.models.ai_guard import GuardChatCompletionsResponse
from pydantic import BaseModel
from strands.models.model import Model
from strands.types.content import ContentBlock, Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

from ._exceptions import AIGuardBlockedError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Configuration TypedDict
# ---------------------------------------------------------------------------


class AIGuardConfig(TypedDict, total=False):
    """AIDR metadata forwarded on every ``guard_chat_completions`` call.

    All fields are optional.  ``llm_provider`` and ``model`` are populated
    automatically from the downstream model's config when not supplied.
    """

    app_id: str
    """Identifier of the source application / agent."""

    collector_instance_id: str
    """AIDR collector instance identifier."""

    user_id: str
    """Identifier of the user or entity initiating the AI interaction."""

    tenant_id: str
    """Tenant identifier for multi-tenant applications."""

    llm_provider: str
    """Name of the LLM provider (e.g. ``"anthropic"``, ``"openai"``)."""

    model: str
    """Name of the AI model (e.g. ``"claude-3-5-sonnet"``, ``"gpt-4o"``)."""

    model_version: str
    """Version identifier for the AI model."""

    source_ip: str
    """IP address of the client making the request."""

    source_location: str
    """Geographic location of the request origin (e.g. ``"US-CA"``, ``"EU-FR"``)."""

    extra_info: dict[str, Any]
    """Additional metadata for AIDR logging in key-value pairs."""


# ---------------------------------------------------------------------------
# Message / tool format converters
# ---------------------------------------------------------------------------


def _strands_messages_to_openai(
    messages: Messages,
    system_prompt: str | None = None,
    system_prompt_content: list[SystemContentBlock] | None = None,
) -> list[dict[str, Any]]:
    """Convert Strands :class:`~strands.types.content.Messages` to the OpenAI-format
    list that AIDR's ``guard_input.messages`` expects.

    Only the fields that AIDR analyses (text, tool calls, tool results) are
    included.  Rich media (images, documents, video) is dropped because AIDR
    does not process binary content.

    ``system_prompt_content`` takes priority over ``system_prompt`` when both are
    provided, matching the Strands SDK convention.
    """
    result: list[dict[str, Any]] = []

    # Prefer structured system_prompt_content over plain system_prompt.
    if system_prompt_content:
        system_text = " ".join(b["text"] for b in system_prompt_content if "text" in b)
        if system_text:
            result.append({"role": "system", "content": system_text})
    elif system_prompt:
        result.append({"role": "system", "content": system_prompt})

    for msg in messages:
        role = msg["role"]
        content_blocks: list[ContentBlock] = msg.get("content", [])

        # Collect text fragments, tool calls, and tool results separately.
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    {
                        "type": "function",
                        "id": tu["toolUseId"],
                        "function": {
                            "name": tu["name"],
                            "arguments": json.dumps(tu.get("input", {})),
                        },
                    }
                )
            elif "toolResult" in block:
                tr = block["toolResult"]
                content_str = ""
                for item in tr.get("content", []):
                    if "text" in item:
                        content_str += item["text"]
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr["toolUseId"],
                        "content": content_str,
                    }
                )

        # Emit the main message first (preserving correct ordering in the conversation),
        # then any tool result messages that follow from it.
        oai_msg: dict[str, Any] = {"role": role}
        if text_parts:
            oai_msg["content"] = "".join(text_parts)
        if tool_calls:
            oai_msg["tool_calls"] = tool_calls
        if text_parts or tool_calls:
            result.append(oai_msg)

        for tr_msg in tool_results:
            result.append(tr_msg)

    return result


def _tool_specs_to_openai(tool_specs: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert Strands :class:`~strands.types.tools.ToolSpec` to OpenAI function format."""
    tools = []
    for spec in tool_specs:
        schema = spec.get("inputSchema", {})
        # inputSchema can be {"json": {...}} (Bedrock-style) or a raw JSON Schema dict.
        if "json" in schema:
            schema = schema["json"]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": schema,
                },
            }
        )
    return tools


# ---------------------------------------------------------------------------
# StreamEvent reconstruction helpers
# ---------------------------------------------------------------------------


def _reconstruct_assistant_message(events: list[StreamEvent]) -> dict[str, Any]:
    """Reconstruct an OpenAI-format assistant message from buffered StreamEvents.

    Returns a dict with ``role="assistant"`` and ``content`` / ``tool_calls``
    populated from the event stream.  This is sent to AIDR as the assistant
    turn in the output-guard call.
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    current_tool: dict[str, Any] | None = None
    current_tool_input_parts: list[str] = []

    for event in events:
        if "contentBlockStart" in event:
            cbs = event["contentBlockStart"]
            start = cbs.get("start", {}) if cbs else {}
            if start and "toolUse" in start and start["toolUse"]:
                tu_start = start["toolUse"]
                current_tool = {
                    "type": "function",
                    "id": tu_start.get("toolUseId", ""),
                    "function": {
                        "name": tu_start.get("name", ""),
                        "arguments": "",
                    },
                }
                current_tool_input_parts = []
            else:
                current_tool = None
                current_tool_input_parts = []

        elif "contentBlockDelta" in event:
            cbd = event["contentBlockDelta"]
            delta = cbd.get("delta", {}) if cbd else {}
            if "text" in delta:
                text_parts.append(delta["text"])
            elif "toolUse" in delta and current_tool is not None:
                current_tool_input_parts.append(delta["toolUse"].get("input", ""))

        elif "contentBlockStop" in event:
            if current_tool is not None:
                current_tool["function"]["arguments"] = "".join(current_tool_input_parts)
                tool_calls.append(current_tool)
                current_tool = None
                current_tool_input_parts = []

    msg: dict[str, Any] = {"role": "assistant"}
    if text_parts:
        msg["content"] = "".join(text_parts)
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _rebuild_events_from_guard_output(
    original_events: list[StreamEvent],
    guard_output: dict[str, Any],
) -> list[StreamEvent]:
    """Rebuild a StreamEvent list whose text content is replaced by AIDR's ``guard_output``.

    Non-text events (tool calls, metadata, message start/stop) are preserved
    from *original_events*.  Text content blocks are replaced with the
    assistant ``content`` field from ``guard_output.messages``.
    """
    messages = guard_output.get("messages", [])
    # Find the last assistant message in guard_output.
    replacement_text: str | None = None
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            replacement_text = m["content"]
            break

    if replacement_text is None:
        # Nothing to replace — return original events unchanged.
        return original_events

    # Rebuild events: keep all non-text-delta events; substitute text deltas.
    rebuilt: list[StreamEvent] = []
    text_block_active = False
    text_emitted = False

    for event in original_events:
        if "contentBlockStart" in event:
            cbs = event["contentBlockStart"]
            start = cbs.get("start", {}) if cbs else {}
            if start and "toolUse" in start and start["toolUse"]:
                text_block_active = False
                rebuilt.append(event)
            else:
                text_block_active = True
                text_emitted = False  # reset per block so each block gets replacement text
                rebuilt.append(event)

        elif "contentBlockDelta" in event and text_block_active:
            cbd = event["contentBlockDelta"]
            delta = cbd.get("delta", {}) if cbd else {}
            if "text" in delta and not text_emitted:
                # Emit the replacement text as a single delta.
                rebuilt.append({"contentBlockDelta": {"delta": {"text": replacement_text}}})
                text_emitted = True
            elif "text" in delta:
                # Skip subsequent text deltas — already emitted replacement.
                pass
            else:
                rebuilt.append(event)

        elif "contentBlockStop" in event:
            text_block_active = False
            rebuilt.append(event)

        else:
            rebuilt.append(event)

    return rebuilt


# ---------------------------------------------------------------------------
# AIGuardModel
# ---------------------------------------------------------------------------


class AIGuardModel(Model):
    """A Strands :class:`~strands.models.Model` wrapper that adds CrowdStrike AIDR
    security guardrails around any downstream model provider.

    Example usage::

        from strands import Agent
        from strands.models import BedrockModel
        from strands_aidr import AIGuardModel

        downstream = BedrockModel(model_id="us.amazon.nova-lite-v1:0")
        model = AIGuardModel(downstream, guard_input=True, guard_output=True)
        agent = Agent(model=model)
        response = agent("Hello!")

    Args:
        downstream: Any Strands ``Model`` instance that performs the actual
            LLM inference.
        base_url_template: AIDR base URL template.  Falls back to the
            ``CS_AIDR_BASE_URL`` environment variable when omitted.
        token: AIDR bearer token.  Falls back to the ``CS_AIDR_TOKEN``
            environment variable when omitted.
        guard_input: Whether to run the AIDR input guard before forwarding
            messages to the downstream model.  Defaults to ``True``.
        guard_output: Whether to run the AIDR output guard after the downstream
            model responds.  Defaults to ``True``.
        **aidr_config: Optional :class:`AIGuardConfig` metadata forwarded on
            every ``guard_chat_completions`` call (``app_id``, ``user_id``, etc.).
    """

    def __init__(
        self,
        downstream: Model,
        *,
        base_url_template: str | None = None,
        token: str | None = None,
        guard_input: bool = True,
        guard_output: bool = True,
        **aidr_config: Unpack[AIGuardConfig],
    ) -> None:
        resolved_base_url = base_url_template or os.environ.get(
            "CS_AIDR_BASE_URL", "https://api.crowdstrike.com/aidr/{SERVICE_NAME}"
        )
        resolved_token = token or os.environ.get("CS_AIDR_TOKEN", "")
        if not resolved_token:
            raise ValueError(
                "AIDR token is required.  Pass token= or set the CS_AIDR_TOKEN environment variable."
            )

        self.downstream = downstream
        self.guard_input = guard_input
        self.guard_output = guard_output
        self._config: dict[str, Any] = dict(aidr_config)
        self._aidr = AIGuard(base_url_template=resolved_base_url, token=resolved_token)

        logger.debug("AIGuardModel initialised | guard_input=%s guard_output=%s", guard_input, guard_output)

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    @override
    def get_config(self) -> AIGuardConfig:
        """Return the current AIDR metadata config."""
        return self._config  # type: ignore[return-value]

    @override
    def update_config(self, **model_config: Unpack[AIGuardConfig]) -> None:  # type: ignore[override]
        """Merge *model_config* into the AIDR metadata config."""
        self._config.update(model_config)

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a conversation through AIDR guards and the downstream model.

        Yields:
            :class:`~strands.types.streaming.StreamEvent` items from the downstream
            model (possibly with content replaced by AIDR's transformed output).

        Raises:
            AIGuardBlockedError: If AIDR blocks either the input or the output.
        """
        # ----------------------------------------------------------
        # 1. Input guard
        # ----------------------------------------------------------
        if self.guard_input:
            messages = await self._run_input_guard(messages, tool_specs, system_prompt, system_prompt_content)

        # ----------------------------------------------------------
        # 2. Downstream model — buffer all events so we can inspect
        #    the complete response before applying the output guard.
        # ----------------------------------------------------------
        buffered_events: list[StreamEvent] = []
        async for event in self.downstream.stream(
            messages,
            tool_specs,
            system_prompt,
            tool_choice=tool_choice,
            system_prompt_content=system_prompt_content,
            invocation_state=invocation_state,
            **kwargs,
        ):
            buffered_events.append(event)

        # ----------------------------------------------------------
        # 3. Output guard
        # ----------------------------------------------------------
        if self.guard_output:
            buffered_events = await self._run_output_guard(
                messages, tool_specs, system_prompt, buffered_events, system_prompt_content
            )

        # ----------------------------------------------------------
        # 4. Yield (possibly transformed) events
        # ----------------------------------------------------------
        for event in buffered_events:
            yield event

    @override
    async def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Delegate structured output to the downstream model with input guarding.

        The input is guarded before forwarding.  Output guarding is not applied
        because ``structured_output`` returns a Pydantic model, not free-form text.
        """
        if self.guard_input:
            prompt = await self._run_input_guard(prompt, None, system_prompt)

        async for item in self.downstream.structured_output(output_model, prompt, system_prompt, **kwargs):
            yield item

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_aidr_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for :meth:`AIGuard.guard_chat_completions`
        from the stored config, omitting keys with ``None`` / empty values."""
        mapping: dict[str, str] = {
            "app_id": "app_id",
            "collector_instance_id": "collector_instance_id",
            "user_id": "user_id",
            "tenant_id": "tenant_id",
            "llm_provider": "llm_provider",
            "model": "model",
            "model_version": "model_version",
            "source_ip": "source_ip",
            "source_location": "source_location",
            "extra_info": "extra_info",
        }
        kwargs: dict[str, Any] = {}
        for key, aidr_key in mapping.items():
            value = self._config.get(key)
            if value is not None:
                kwargs[aidr_key] = value
        return kwargs

    def _call_aidr_sync(
        self,
        event_type: str,
        guard_input_payload: dict[str, Any],
    ) -> GuardChatCompletionsResponse | None:
        """Synchronous AIDR call (run in executor to avoid blocking the event loop)."""
        extra_kwargs = self._build_aidr_kwargs()
        response = self._aidr.guard_chat_completions(
            guard_input=guard_input_payload,
            event_type=event_type,  # type: ignore[arg-type]
            **extra_kwargs,
        )
        return response if response else None

    async def _call_aidr(
        self,
        event_type: str,
        guard_input_payload: dict[str, Any],
    ) -> GuardChatCompletionsResponse | None:
        """Async wrapper that runs the synchronous AIDR client in a thread-pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._call_aidr_sync,
            event_type,
            guard_input_payload,
        )

    async def _run_input_guard(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None,
        system_prompt: str | None,
        system_prompt_content: list[SystemContentBlock] | None = None,
    ) -> Messages:
        """Run the AIDR input guard.  Returns (possibly substituted) messages."""
        payload: dict[str, Any] = {
            "messages": _strands_messages_to_openai(messages, system_prompt, system_prompt_content),
        }
        if tool_specs:
            payload["tools"] = _tool_specs_to_openai(tool_specs)

        logger.debug("AIDR input guard | message_count=%d", len(messages))
        response = await self._call_aidr("input", payload)

        if response is None:
            logger.warning("AIDR input guard returned no result; proceeding without guard")
            return messages

        result = response.result
        if result is None:
            logger.warning("AIDR input guard result was None; proceeding without guard")
            return messages

        if result.blocked:
            logger.info("AIDR input guard BLOCKED | summary=%s", response.summary)
            raise AIGuardBlockedError("input", response.summary or "")

        if result.transformed and result.guard_output:
            messages = _substitute_messages(messages, result.guard_output, system_prompt, system_prompt_content)
            logger.debug("AIDR input guard transformed messages")

        return messages

    async def _run_output_guard(
        self,
        input_messages: Messages,
        tool_specs: list[ToolSpec] | None,
        system_prompt: str | None,
        buffered_events: list[StreamEvent],
        system_prompt_content: list[SystemContentBlock] | None = None,
    ) -> list[StreamEvent]:
        """Run the AIDR output guard.  Returns (possibly rebuilt) StreamEvents."""
        assistant_oai = _reconstruct_assistant_message(buffered_events)

        # Build a message list that includes the full conversation + assistant response.
        conversation_oai = _strands_messages_to_openai(input_messages, system_prompt, system_prompt_content)
        conversation_oai.append(assistant_oai)

        payload: dict[str, Any] = {"messages": conversation_oai}
        if tool_specs:
            payload["tools"] = _tool_specs_to_openai(tool_specs)

        logger.debug("AIDR output guard | conversation_length=%d", len(conversation_oai))
        response = await self._call_aidr("output", payload)

        if response is None:
            logger.warning("AIDR output guard returned no result; returning original events")
            return buffered_events

        result = response.result
        if result is None:
            logger.warning("AIDR output guard result was None; returning original events")
            return buffered_events

        if result.blocked:
            logger.info("AIDR output guard BLOCKED | summary=%s", response.summary)
            raise AIGuardBlockedError("output", response.summary or "")

        if result.transformed and result.guard_output:
            buffered_events = _rebuild_events_from_guard_output(buffered_events, result.guard_output)
            logger.debug("AIDR output guard transformed response")

        return buffered_events


# ---------------------------------------------------------------------------
# Private helper: substitute Strands messages from AIDR guard_output
# ---------------------------------------------------------------------------


def _substitute_messages(
    original: Messages,
    guard_output: dict[str, Any],
    system_prompt: str | None = None,
    system_prompt_content: list[SystemContentBlock] | None = None,
) -> Messages:
    """Return a copy of *original* with text content replaced from *guard_output*.

    The message structure (roles, order, tool calls/results) is preserved.
    ``system_prompt`` / ``system_prompt_content`` must be passed when the same
    values were used to build the AIDR payload, so the index offset introduced
    by the prepended system entry is correctly accounted for.
    """
    updated_oai = guard_output.get("messages", [])
    # Build a lookup: index → new content string (only for messages with a
    # ``content`` field, i.e. non-tool-call messages).
    content_by_index: dict[int, str] = {}
    for idx, m in enumerate(updated_oai):
        if "content" in m and isinstance(m["content"], str):
            content_by_index[idx] = m["content"]

    # When _strands_messages_to_openai prepended a system entry, the OAI list
    # is offset by 1 relative to the original Strands messages list.
    has_system = bool(system_prompt_content) or bool(system_prompt)

    result: Messages = []
    oai_idx = 1 if has_system else 0  # skip the system entry if present
    for msg in original:
        role = msg["role"]
        new_blocks: list[ContentBlock] = []
        for block in msg.get("content", []):
            if "text" in block:
                replacement = content_by_index.get(oai_idx, block.get("text", ""))
                new_blocks.append({"text": replacement})
                oai_idx += 1
            else:
                new_blocks.append(block)
                if "toolUse" in block or "toolResult" in block:
                    oai_idx += 1  # account for corresponding OAI message
        result.append({"role": role, "content": new_blocks})

    return result
