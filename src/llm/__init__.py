"""Pick a model backend from the environment.

Configuration is environment, not code, so that pointing the pipeline at a
different model is a deployment change. With nothing set at all the choice
falls to whatever there is a key for, and finally to a model on this machine,
which is the one option that needs no account.

    ROOTCITE_LLM_PROVIDER   claude | openai | local
    ROOTCITE_LLM_MODEL      overrides the backend's default model
    ROOTCITE_LLM_BASE_URL   overrides where an OpenAI-compatible server listens
    ANTHROPIC_API_KEY       read by the Anthropic SDK
    OPENAI_API_KEY          read by the OpenAI SDK
    ROOTCITE_LOCAL_API_KEY  only if your local runtime was started with one
"""

from __future__ import annotations

import logging
import os

from llm.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Provider,
    ProviderConfig,
    ProviderError,
    RefusalError,
    checked_base_url,
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

logger = logging.getLogger(__name__)

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
    """Build the configured backend. Each key is read by the SDK it belongs to."""
    chosen = name or configured_name()
    if chosen not in PROVIDERS:
        raise ValueError(f"unknown provider {chosen!r}; try one of {', '.join(sorted(PROVIDERS))}")

    provider = PROVIDERS[chosen]
    return provider(
        ProviderConfig(
            model=model or os.environ.get("ROOTCITE_LLM_MODEL") or provider.default_model,
            api_key=os.environ.get(provider.key_variable),
            base_url=_endpoint(provider, base_url),
            max_tokens=max_tokens,
        )
    )


def _endpoint(provider: type[Provider], asked: str | None) -> str | None:
    """Where to send requests, for the backends that can be pointed anywhere."""
    if not provider.accepts_base_url:
        if asked:
            raise ValueError(f"{provider.name} talks to its own API and takes no base url")
        if os.environ.get("ROOTCITE_LLM_BASE_URL"):
            logger.warning("ROOTCITE_LLM_BASE_URL does not apply to %s; ignoring it", provider.name)
        return None

    chosen = asked or os.environ.get("ROOTCITE_LLM_BASE_URL")
    return checked_base_url(chosen) if chosen else provider.default_base_url
