"""Claude on Amazon Bedrock through the Converse API, with no Anthropic key.

spiritwriter's AnthropicProvider needs either the Claude Agent SDK or an
ANTHROPIC_API_KEY. This provider calls bedrock-runtime directly with the
standard AWS credential chain, as BedrockEmbedder does.
"""

from __future__ import annotations

import asyncio
from typing import Any

from spiritwriter.llm import LLMProvider

from .embed import bedrock_runtime_client

DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-5-5"


class BedrockLLMProvider(LLMProvider):
    def __init__(
        self,
        model: str = DEFAULT_BEDROCK_MODEL,
        region: str | None = None,
        max_tokens: int = 8192,
        client=None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._region = region
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = bedrock_runtime_client(self._region)
        return self._client

    def _converse(self, prompt: str, model: str) -> tuple[str, dict[str, int]]:
        resp = self.client.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens},
        )
        text = "".join(
            block.get("text", "") for block in resp["output"]["message"]["content"]
        ).strip()
        if not text:
            raise RuntimeError(f"Bedrock returned no text (stopReason={resp.get('stopReason')})")
        u = resp.get("usage", {})
        usage = {
            "input_tokens": u.get("inputTokens", 0),
            "output_tokens": u.get("outputTokens", 0),
            "total_tokens": u.get("totalTokens", 0),
        }
        return text, usage

    async def query(self, prompt: str, *, return_usage: bool = False, **kwargs: Any) -> Any:
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        text, usage = await asyncio.to_thread(
            self._converse, prompt, kwargs.get("model") or self.model
        )
        return (text, usage) if return_usage else text

    async def query_with_image(self, prompt: str, image_data: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError("BedrockLLMProvider is text-only")

    async def query_with_images(self, prompt: str, images: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("BedrockLLMProvider is text-only")
