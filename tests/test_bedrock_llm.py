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


def test_system_prompt_goes_in_the_system_field():
    calls = []

    class Fake(FakeConverse):
        def converse(self, **request):
            calls.append(request)
            return super().converse(request["modelId"], request["messages"], request["inferenceConfig"])

    asyncio.run(BedrockLLMProvider(client=Fake()).query("hi", system_prompt="be brief"))
    assert calls[0]["system"] == [{"text": "be brief"}]
    assert calls[0]["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]


class FlakyConverse(FakeConverse):
    """First reply empty, second cut off at max_tokens, third good."""

    def __init__(self):
        super().__init__()
        self.replies = [("", "end_turn"), ('{"ok": ', "max_tokens"), ('{"ok": 2}', "end_turn")]

    def converse(self, modelId, messages, inferenceConfig):
        text, stop = self.replies.pop(0)
        self.calls.append(modelId)
        return {"output": {"message": {"content": [{"text": text}]}}, "stopReason": stop, "usage": {}}


def test_query_json_retries_empty_and_truncated_replies():
    fake = FlakyConverse()
    assert asyncio.run(_query_json(BedrockLLMProvider(client=fake), "p", None)) == {"ok": 2}
    assert len(fake.calls) == 3


def test_rejects_anthropic_api_model_id():
    with pytest.raises(ValueError, match="us.anthropic.claude-sonnet-5-5"):
        BedrockLLMProvider("claude-sonnet-5-5", client=FakeConverse())
    with pytest.raises(ValueError):
        asyncio.run(BedrockLLMProvider(client=FakeConverse()).query("hi", model="claude-haiku-5-5"))


def test_cli_rejects_anthropic_model_id_with_llm_bedrock(capsys):
    from semantic_ruler.demo import main

    with pytest.raises(SystemExit) as exc:
        main(["--llm", "bedrock", "--model", "claude-sonnet-5-5", "--no-plot"])
    assert exc.value.code == 2
    assert "not a Bedrock model ID" in capsys.readouterr().err
