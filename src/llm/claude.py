"""Claude, through the Anthropic SDK.

Thinking is left adaptive: deciding which of a paper's ancestors actually
carried an idea is not a lookup. Refusal fallbacks are on, so a declined
request is re-run server-side on Anthropic's recommended substitute rather
than surfacing as an error the narrative layer would have to handle.
"""

from __future__ import annotations

from typing import Any

import anthropic

from llm.base import Completion, Provider, ProviderConfig, ProviderError, RefusalError

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeProvider(Provider):
    name = "claude"
    default_model = "claude-opus-5"
    key_variable = "ANTHROPIC_API_KEY"

    def __init__(self, config: ProviderConfig, client: Any | None = None) -> None:
        super().__init__(config)
        self.client = client or anthropic.Anthropic(api_key=config.api_key)

    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        return self._message(prompt, system)

    def answer_as(self, prompt: str, schema: dict, *, system: str | None = None) -> Completion:
        return self._message(
            prompt, system, output_config={"format": {"type": "json_schema", "schema": schema}}
        )

    def _message(self, prompt: str, system: str | None, **extra: Any) -> Completion:
        try:
            response = self.client.beta.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                thinking={"type": "adaptive"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
                **({"system": system} if system else {}),
                **extra,
            )
        except anthropic.AnthropicError as failure:
            raise ProviderError(f"{self.name} / {self.config.model}: {failure}") from failure
        if response.stop_reason == "refusal":
            raise RefusalError(f"{self.config.model} declined to answer")
        return Completion(
            text="".join(block.text for block in response.content if block.type == "text"),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
