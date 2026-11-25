"""BedrockLLMProvider and _query_json error handling: no network, no AWS."""

import asyncio

import pytest

from semantic_ruler.generate import _query_json
from semantic_ruler.llm_bedrock import DEFAULT_BEDROCK_MODEL, BedrockLLMProvider


class FakeConverse:
    def __init__(self, text='{"ok": 1}'):
        self.text = text
        self.calls = []

    def converse(self, modelId, messages, inferenceConfig):
        self.calls.append((modelId, messages, inferenceConfig))
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": self.text}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
        }


def test_query_returns_text_and_usage():
    fake = FakeConverse("hello")
    llm = BedrockLLMProvider(client=fake)
    text, usage = asyncio.run(llm.query("hi", return_usage=True))
    assert text == "hello"
    assert usage == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
    model, messages, cfg = fake.calls[0]
    assert model == DEFAULT_BEDROCK_MODEL
    assert messages == [{"role": "user", "content": [{"text": "hi"}]}]
    assert cfg["maxTokens"] == 8192


def test_model_kwarg_overrides_default():
    fake = FakeConverse()
    asyncio.run(BedrockLLMProvider(client=fake).query("hi", model="us.anthropic.claude-haiku-5-5"))
    assert fake.calls[0][0] == "us.anthropic.claude-haiku-5-5"


def test_query_json_parses_bedrock_reply():
    assert asyncio.run(_query_json(BedrockLLMProvider(client=FakeConverse()), "p", None)) == {"ok": 1}


def test_query_json_does_not_retry_provider_errors():
    calls = []

    class Broken:
        async def query(self, prompt, **kw):
            calls.append(prompt)
            raise ValueError("ANTHROPIC_API_KEY not set")

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        asyncio.run(_query_json(Broken(), "p", None))
    assert len(calls) == 1
