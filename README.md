# aidr-strands-python

CrowdStrike AIDR security guardrails for the [Strands Agents SDK](https://strandsagents.com/).

## Overview

`strands-aidr` provides `AIGuardModel`, a wrapper that intercepts any Strands
model provider and runs CrowdStrike AIDR analysis on both:

- **Input** — user messages and tool definitions, before they reach the LLM.
- **Output** — the LLM's response, before it is returned to the agent.

AIDR can detect and block malicious prompts, PII, secrets, MCP tool injection,
policy violations, and more.  Blocked requests raise `AIGuardBlockedError`.
Transformed (redacted) content is automatically substituted in the message
stream so the downstream model and agent only ever see clean data.

## Installation

```bash
pip install strands-aidr
```

## Requirements

- Python 3.12+
- A CrowdStrike AIDR [Application collector](https://falcon.crowdstrike.com/aidr/) token
- Any Strands Agents SDK model provider (Bedrock, OpenAI, Anthropic, …)

## Quick start

```python
from strands import Agent
from strands.models import BedrockModel
from strands_aidr import AIGuardModel

downstream = BedrockModel(model_id="us.amazon.nova-lite-v1:0")

model = AIGuardModel(
    downstream,
    guard_input=True,
    guard_output=True,
)

agent = Agent(model=model)
response = agent("Hello!")
```

Credentials are read from environment variables by default:

```bash
export CS_AIDR_BASE_URL="https://api.crowdstrike.com/aidr/{SERVICE_NAME}"
export CS_AIDR_TOKEN="pts_..."
```

Or passed directly:

```python
model = AIGuardModel(
    downstream,
    base_url_template="https://api.crowdstrike.com/aidr/{SERVICE_NAME}",
    token="pts_...",
)
```

## Configuration

`AIGuardModel` accepts optional AIDR metadata that appears in Findings logs:

```python
model = AIGuardModel(
    downstream,
    app_id="hr-portal",
    user_id="mary.potter",
    tenant_id="acme-corp",
    extra_info={"app_name": "HR Portal", "app_version": "1.2.3"},
)
```

## Handling blocked requests

```python
from strands_aidr import AIGuardBlockedError

try:
    response = agent("Please ignore all previous instructions and…")
except AIGuardBlockedError as e:
    print(f"Blocked at {e.event_type}: {e.summary}")
```

## Architecture

```
Agent
  └── AIGuardModel.stream()
        ├── [guard_input=True]  AIDR guard_chat_completions(event_type="input")
        │      blocked=True  → raise AIGuardBlockedError("input", …)
        │      transformed   → substitute redacted messages
        │
        ├── downstream.stream()   ← buffered (collected in memory)
        │
        └── [guard_output=True] AIDR guard_chat_completions(event_type="output")
               blocked=True  → raise AIGuardBlockedError("output", …)
               transformed   → rebuild StreamEvents with redacted text
```

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/test_model.py -v
```

Integration tests require real credentials and are skipped automatically when
`CS_AIDR_TOKEN` / `CS_AIDR_BASE_URL` are not set:

```bash
CS_AIDR_TOKEN=pts_... CS_AIDR_BASE_URL=... pytest tests/test_integration.py -v
```
