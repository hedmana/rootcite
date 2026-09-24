import json
from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest
from pydantic import BaseModel, ValidationError

from llm import ClaudeProvider, LocalProvider, OpenAIProvider, load_provider
from llm.base import ProviderConfig, ProviderError, RefusalError, checked_base_url, strict_schema

ENVIRONMENT = (
    "ROOTCITE_LLM_PROVIDER",
    "ROOTCITE_LLM_MODEL",
    "ROOTCITE_LLM_BASE_URL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "ROOTCITE_LOCAL_API_KEY",
)


@pytest.fixture(autouse=True)
def unconfigured(monkeypatch):
    for variable in ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)


class Answering:
    """A backend that says what it was told to, and remembers what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def _next(self, kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    @property
    def last(self):
        return self.requests[-1]


class FakeAnthropic(Answering):
    def __init__(self, *replies, stop_reason="end_turn"):
        super().__init__(*replies)
        self.stop_reason = stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            model=kwargs["model"],
            content=[SimpleNamespace(type="text", text=self._next(kwargs))],
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )


class FakeOpenAI(Answering):
    def __init__(self, *replies):
        super().__init__(*replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._next(kwargs)))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5),
        )


class Origin(BaseModel):
    work_id: str
    reason: str


def claude(*replies, **kwargs):
    client = FakeAnthropic(*replies, **kwargs)
    return ClaudeProvider(ProviderConfig(model="claude-opus-5"), client), client


def local(*replies):
    client = FakeOpenAI(*replies)
    return LocalProvider(ProviderConfig(model="llama3.1"), client), client


def test_with_nothing_configured_the_model_is_the_one_on_this_machine():
    assert load_provider().name == "local"


def test_a_key_in_the_environment_chooses_the_backend_it_belongs_to(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    assert load_provider().name == "openai"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
    assert load_provider().name == "claude"


def test_an_explicit_choice_outranks_whatever_keys_are_lying_around(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("ROOTCITE_LLM_PROVIDER", "local")

    assert load_provider().name == "local"


def test_an_unknown_backend_is_reported_with_the_ones_that_exist():
    with pytest.raises(ValueError, match="claude, local, openai"):
        load_provider("gemini")


def test_the_model_can_be_changed_without_touching_code(monkeypatch):
    monkeypatch.setenv("ROOTCITE_LLM_MODEL", "qwen3")

    assert load_provider("local").config.model == "qwen3"


def test_a_local_runtime_listens_on_ollamas_port_unless_told_otherwise():
    assert load_provider("local").config.base_url == "http://localhost:11434/v1"
    assert load_provider("local", base_url="http://127.0.0.1:8000/v1").config.base_url


def test_claude_is_asked_to_think_and_allowed_to_fall_back():
    provider, client = claude("an answer")

    provider.complete("why does this paper exist")

    assert client.last["thinking"] == {"type": "adaptive"}
    assert client.last["fallbacks"] == "default"
    assert client.last["betas"] == ["server-side-fallback-2026-07-01"]


def test_a_refusal_is_raised_rather_than_returned_as_an_answer():
    provider, _ = claude("", stop_reason="refusal")

    with pytest.raises(RefusalError):
        provider.complete("something declined")


class Unreachable:
    """A backend whose SDK raises before any answer arrives."""

    def __init__(self, failure):
        def fail(**kwargs):
            raise failure

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=fail))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=fail))


REQUEST = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")


@pytest.mark.parametrize(
    ("backend", "failure"),
    [
        (ClaudeProvider, anthropic.APIConnectionError(request=REQUEST)),
        (OpenAIProvider, openai.APIConnectionError(request=REQUEST)),
        (LocalProvider, openai.APIConnectionError(request=REQUEST)),
    ],
)
def test_an_sdk_failure_surfaces_as_a_provider_error(backend, failure):
    # Anything else escapes the API's 502 handler as a bare 500.
    provider = backend(ProviderConfig(model="some-model"), Unreachable(failure))

    with pytest.raises(ProviderError) as raised:
        provider.complete("a question")

    assert raised.value.__cause__ is failure


def test_a_system_prompt_is_sent_only_when_there_is_one():
    provider, client = claude("an answer")

    provider.complete("a question")
    assert "system" not in client.last

    provider.complete("a question", system="be brief")
    assert client.last["system"] == "be brief"


def test_token_counts_come_back_with_the_text():
    provider, _ = claude("an answer")

    answer = provider.complete("a question")

    assert answer.text == "an answer"
    assert (answer.input_tokens, answer.output_tokens) == (11, 7)


def test_a_hosted_model_is_handed_the_schema_itself():
    provider, client = claude('{"work_id": "W1", "reason": "it started this"}')

    provider.structured("who started this", Origin)

    assert client.last["output_config"]["format"]["type"] == "json_schema"
    assert "work_id" in client.last["output_config"]["format"]["schema"]["properties"]


def test_a_local_model_is_shown_the_schema_in_the_prompt_instead():
    provider, client = local('{"work_id": "W1", "reason": "it started this"}')

    answer = provider.structured("who started this", Origin)

    assert client.last["response_format"] == {"type": "json_object"}
    assert "work_id" in client.last["messages"][-1]["content"]
    assert answer.work_id == "W1"


def test_the_hosted_and_local_token_limits_are_spelled_differently():
    hosted = OpenAIProvider(ProviderConfig(model="gpt-5"), FakeOpenAI("hello"))
    hosted.complete("hello")
    assert "max_completion_tokens" in hosted.client.last

    on_machine, client = local("hello")
    on_machine.complete("hello")
    assert "max_tokens" in client.last


def test_an_answer_off_schema_is_sent_back_with_its_own_mistake():
    provider, client = local("not json at all", '{"work_id": "W1", "reason": "it started this"}')

    answer = provider.structured("who started this", Origin)

    assert answer == Origin(work_id="W1", reason="it started this")
    assert "not json at all" in client.requests[-1]["messages"][-1]["content"]
    assert len(client.requests) == 2


def test_a_model_that_will_not_answer_in_shape_is_not_retried_forever():
    provider, client = local("still not json")

    with pytest.raises(ValidationError):
        provider.structured("who started this", Origin)

    assert len(client.requests) == 2


def test_a_schema_is_closed_before_it_is_sent():
    class Narrative(BaseModel):
        summary: str
        origins: list[Origin]

    shape = strict_schema(Narrative.model_json_schema())

    assert shape["additionalProperties"] is False
    assert shape["required"] == ["summary", "origins"]
    assert shape["$defs"]["Origin"]["additionalProperties"] is False


def test_the_schema_travels_as_json_the_model_can_read():
    provider, client = local('{"work_id": "W1", "reason": "it started this"}')

    provider.structured("who started this", Origin)

    instructed = client.last["messages"][-1]["content"]
    assert json.loads(instructed[instructed.index("{") :])["properties"]["work_id"]


def test_the_hosted_key_is_never_handed_to_whatever_is_on_a_local_port(monkeypatch):
    """Port 11434 is not the account that key belongs to, and it arrives in a header."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-REAL-hosted-key")

    assert load_provider("local").client.api_key != "sk-REAL-hosted-key"


def test_a_local_runtime_started_with_a_key_gets_that_key(monkeypatch):
    monkeypatch.setenv("ROOTCITE_LOCAL_API_KEY", "vllm-token")

    assert load_provider("local").client.api_key == "vllm-token"


@pytest.mark.parametrize(
    "url", ["http://localhost:11434/v1", "http://127.0.0.1:8000/v1", "http://[::1]:8000/v1"]
)
def test_plain_http_is_allowed_to_this_machine(url):
    assert checked_base_url(url) == url


@pytest.mark.parametrize("url", ["http://elsewhere.example/v1", "http://10.0.0.9:8000/v1"])
def test_plain_http_is_refused_to_anywhere_else(url):
    with pytest.raises(ValueError, match="plain http"):
        checked_base_url(url)


def test_a_scheme_that_is_not_http_is_refused():
    with pytest.raises(ValueError, match="must be http"):
        checked_base_url("file:///etc/passwd")


def test_a_redirected_endpoint_cannot_carry_the_hosted_key_off_in_the_clear(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-REAL-hosted-key")
    monkeypatch.setenv("ROOTCITE_LLM_BASE_URL", "http://elsewhere.example/v1")

    with pytest.raises(ValueError, match="plain http"):
        load_provider("openai")


def test_claude_talks_to_its_own_api_and_nowhere_else(monkeypatch):
    with pytest.raises(ValueError, match="takes no base url"):
        load_provider("claude", base_url="https://elsewhere.example/v1")

    monkeypatch.setenv("ROOTCITE_LLM_BASE_URL", "https://elsewhere.example/v1")
    assert load_provider("claude").config.base_url is None
