"""Pick a model backend from the environment.

Configuration is environment, not code, so that pointing the pipeline at a
different model is a deployment change. With nothing set at all the choice
falls to whatever there is a key for, and finally to a model on this machine,
which is the one option that needs no account.

    ROOTCITE_LLM_PROVIDER   claude | openai | local
    ROOTCITE_LLM_MODEL      overrides the backend's default model
    ROOTCITE_LLM_BASE_URL   overrides where a local runtime is listening
    ANTHROPIC_API_KEY       read by the Anthropic SDK
    OPENAI_API_KEY          read by the OpenAI SDK
"""

from __future__ import annotations

import os

from llm.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Provider,
    ProviderConfig,
    ProviderError,
    RefusalError,
)
from llm.claude import ClaudeProvider
from llm.openai_compatible import LocalProvider, OpenAIProvider

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "PROVIDERS",
    "ClaudeProvider",
    "Completion",
    "LocalProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderConfig",
    "ProviderError",
    "RefusalError",
    "load_provider",
]

PROVIDERS: dict[str, type[Provider]] = {
    provider.name: provider for provider in (ClaudeProvider, OpenAIProvider, LocalProvider)
}


def configured_name() -> str:
    """What was asked for, else what there is a key for, else this machine."""
    if asked := os.environ.get("ROOTCITE_LLM_PROVIDER"):
        return asked
    if os.environ.get(ClaudeProvider.key_variable):
        return ClaudeProvider.name
    if os.environ.get(OpenAIProvider.key_variable):
        return OpenAIProvider.name
    return LocalProvider.name


def load_provider(
    name: str | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Provider:
    """Build the configured backend. Keys are read by the SDKs, never passed around."""
    chosen = name or configured_name()
    if chosen not in PROVIDERS:
        raise ValueError(f"unknown provider {chosen!r}; try one of {', '.join(sorted(PROVIDERS))}")

    provider = PROVIDERS[chosen]
    return provider(
        ProviderConfig(
            model=model or os.environ.get("ROOTCITE_LLM_MODEL") or provider.default_model,
            api_key=os.environ.get(provider.key_variable),
            base_url=base_url
            or os.environ.get("ROOTCITE_LLM_BASE_URL")
            or provider.default_base_url,
            max_tokens=max_tokens,
        )
    )
