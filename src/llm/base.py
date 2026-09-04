"""The contract every model backend has to satisfy.

The narrative layer asks for two things: prose, and an answer shaped like a
schema. Which model produces them is a deployment decision, not a design one,
so everything provider-specific stops at this file.

Structured answers are validated here rather than in each backend. A hosted
model with native schema support and a small local one prompted to behave get
the same treatment: parse, and if the parse fails, show the model its own
mistake once before giving up.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 16000

Schema = TypeVar("Schema", bound=BaseModel)


class ProviderError(RuntimeError):
    """The backend could not answer."""


class RefusalError(ProviderError):
    """The backend declined to answer, rather than failing to."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ProviderConfig:
    model: str
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS


def strict_schema(schema: dict) -> dict:
    """Close every object in a schema, which is what json_schema mode requires.

    Pydantic leaves objects open to extra keys and lists only the fields that
    have no default as required. Both providers reject that.
    """
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        schema["required"] = list(schema.get("properties", {}))
    for nested in (
        *schema.get("properties", {}).values(),
        *schema.get("$defs", {}).values(),
        *([schema["items"]] if isinstance(schema.get("items"), dict) else []),
    ):
        strict_schema(nested)
    return schema


class Provider(ABC):
    """One model backend, reached the way that backend prefers to be reached."""

    name: ClassVar[str]
    default_model: ClassVar[str]
    default_base_url: ClassVar[str | None] = None
    key_variable: ClassVar[str]

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """Prose."""

    @abstractmethod
    def answer_as(self, prompt: str, schema: dict, *, system: str | None = None) -> Completion:
        """The same, constrained to JSON matching `schema`."""

    def structured(self, prompt: str, schema: type[Schema], *, system: str | None = None) -> Schema:
        """An answer parsed into `schema`, with one chance to correct itself."""
        shape = strict_schema(schema.model_json_schema())
        completion = self.answer_as(prompt, shape, system=system)
        try:
            return schema.model_validate_json(completion.text)
        except ValidationError as invalid:
            logger.warning("%s returned an answer off schema; asking again", self.name)
            retry = self.answer_as(
                _corrected(prompt, completion.text, invalid), shape, system=system
            )
            return schema.model_validate_json(retry.text)


def _corrected(prompt: str, answer: str, error: ValidationError) -> str:
    return (
        f"{prompt}\n\nYour previous answer did not fit the schema:\n{answer}\n\n"
        f"It failed validation with:\n{error}\n\nAnswer again, valid this time."
    )


def as_json_instruction(prompt: str, schema: dict) -> str:
    """For backends with no native schema support: say it in the prompt instead."""
    return (
        f"{prompt}\n\nReply with JSON matching this schema, and with nothing else:\n"
        f"{json.dumps(schema)}"
    )
