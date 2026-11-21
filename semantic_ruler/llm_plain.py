"""A text-only LLM provider for prompts that contain untrusted text.

spiritwriter's AnthropicProvider goes through the Claude Agent SDK with its
default options, which leaves the agent's built-in tools available. That is
fine for prompts we write ourselves, but not when the prompt embeds fetched
web pages: this provider runs the same SDK with every tool removed, a single
turn, no project settings, and an empty working directory.
"""

from __future__ import annotations

import tempfile
from typing import Any

from spiritwriter.llm import LLMProvider


class PlainTextProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self._cwd = tempfile.mkdtemp(prefix="semantic-ruler-llm-")
        self.last_tools: list[str] | None = None  # tools the session reported at init

    async def query(self, prompt: str, *, return_usage: bool = False, **kwargs: Any) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions, query

        options = ClaudeAgentOptions(
            model=kwargs.get("model") or self.model,
            tools=[],
            max_turns=1,
            cwd=self._cwd,
        )
        result, assistant, usage = None, [], None
        async for message in query(prompt=prompt, options=options):
            if getattr(message, "subtype", None) == "init" and isinstance(getattr(message, "data", None), dict):
                self.last_tools = list(message.data.get("tools", []))
            elif hasattr(message, "is_error") and hasattr(message, "result"):
                if message.is_error:
                    raise RuntimeError(f"LLM run failed ({message.subtype}): {message.result}")
                result = message.result
                u = message.usage or {}
                usage = {
                    "input_tokens": u.get("input_tokens", 0),
                    "output_tokens": u.get("output_tokens", 0),
                    "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
                }
            elif hasattr(message, "model") and isinstance(getattr(message, "content", None), list):
                assistant.extend(b.text for b in message.content if hasattr(b, "text"))
        text = (result if isinstance(result, str) and result else "\n".join(assistant)).strip()
        return (text, usage) if return_usage else text

    async def query_with_image(self, prompt: str, image_data: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError("PlainTextProvider is text-only")

    async def query_with_images(self, prompt: str, images: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("PlainTextProvider is text-only")
