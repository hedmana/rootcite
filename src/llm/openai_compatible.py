"""GPT, and any model running on your own machine.

These are one implementation because they are one API. Ollama, LM Studio, vLLM
and llama.cpp all serve the OpenAI chat endpoint, so a local model is the same
client pointed at localhost with a key nothing checks.

Where they differ is what they will honour. A hosted model can be handed a
schema and made to obey it; a small local one is asked for JSON and shown the
schema in the prompt, which every runtime supports.
"""

from __future__ import annotations

from typing import Any

import openai

from llm.base import Completion, Provider, ProviderConfig, ProviderError, as_json_instruction

UNCHECKED_KEY = "not-checked-locally"


class OpenAIProvider(Provider):
    name = "openai"
    default_model = "gpt-5"
    key_variable = "OPENAI_API_KEY"
    accepts_base_url = True
    token_parameter = "max_completion_tokens"

    def __init__(self, config: ProviderConfig, client: Any | None = None) -> None:
        super().__init__(config)
        self.client = client or openai.OpenAI(
            api_key=config.api_key or UNCHECKED_KEY, base_url=config.base_url
        )

    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        return self._chat(prompt, system)

    def answer_as(self, prompt: str, schema: dict, *, system: str | None = None) -> Completion:
        return self._chat(
            prompt,
            system,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
        )

    def _chat(self, prompt: str, system: str | None, **extra: Any) -> Completion:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                **{self.token_parameter: self.config.max_tokens},
                **extra,
            )
        except openai.OpenAIError as failure:
            raise ProviderError(f"{self.name} / {self.config.model}: {failure}") from failure
        usage = getattr(response, "usage", None)
        return Completion(
            text=response.choices[0].message.content or "",
            model=response.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class LocalProvider(OpenAIProvider):
    """A model on your own hardware. No key leaves the machine, because there is none."""

    name = "local"
    default_model = "llama3.1"
    default_base_url = "http://localhost:11434/v1"
    # Deliberately not OPENAI_API_KEY. Whatever is listening on a local port is
    # not the account that key belongs to, and it would arrive in a header.
    key_variable = "ROOTCITE_LOCAL_API_KEY"
    token_parameter = "max_tokens"

    def answer_as(self, prompt: str, schema: dict, *, system: str | None = None) -> Completion:
        return self._chat(
            as_json_instruction(prompt, schema), system, response_format={"type": "json_object"}
        )
