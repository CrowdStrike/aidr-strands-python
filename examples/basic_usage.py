"""Basic usage example: wrap a Bedrock model with AIDR guardrails."""

from strands import Agent
from strands.models import BedrockModel

from strands_aidr import AIGuardModel

# Any Strands model provider works as the downstream.
downstream = BedrockModel(model_id="us.amazon.nova-lite-v1:0")

# Wrap it with AIDR.  Credentials are read from CS_AIDR_TOKEN / CS_AIDR_BASE_URL.
model = AIGuardModel(
    downstream,
    guard_input=True,
    guard_output=True,
    # Optional AIDR metadata for logging/policy matching:
    app_id="my-app",
)

agent = Agent(model=model)
response = agent("What is 2 + 2?")
print(response)
