import httpx
import pytest

from graph.openalex import (
    MAX_IDS_PER_FILTER,
    OpenAlexClient,
    OpenAlexError,
    Work,
    abstract_from_inverted_index,
    strip_id_prefix,
)


def work_payload(work_id="W1", **overrides):
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": "Semi-Supervised Classification with Graph Convolutional Networks",
        "publication_year": 2016,
        "publication_date": "2016-09-09",
        "cited_by_count": 8058,
        "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W3"],
        "abstract_inverted_index": {"scalable": [0], "approach": [1], "graphs": [2]},
        "primary_topic": {"id": "https://openalex.org/T11273"},
        "authorships": [
            {"author": {"display_name": "Thomas N. Kipf"}},
            {"author": {"display_name": "Max Welling"}},
        ],
        **overrides,
    }


def client_for(handler, **kwargs):
    return OpenAlexClient(transport=httpx.MockTransport(handler), **kwargs)


def test_strip_id_prefix_is_idempotent():
    assert strip_id_prefix("https://openalex.org/W1") == "W1"
    assert strip_id_prefix("W1") == "W1"


def test_abstract_reconstruction_orders_tokens_by_position():
    index = {"networks": [2], "graph": [0], "neural": [1], "the": [3, 5], "on": [4]}
    assert abstract_from_inverted_index(index) == "graph neural networks the on the"


def test_abstract_reconstruction_handles_missing_index():
    assert abstract_from_inverted_index(None) is None
    assert abstract_from_inverted_index({}) is None


def test_work_flattens_nested_payload():
    work = Work.model_validate(work_payload())

    assert work.id == "W1"
    assert work.referenced_works == ["W2", "W3"]
    assert work.topic_id == "T11273"
    assert work.authors == ["Thomas N. Kipf", "Max Welling"]
    assert work.abstract == "scalable approach graphs"
    assert work.cited_by_count == 8058


def test_work_tolerates_absent_optional_fields():
    work = Work.model_validate({"id": "https://openalex.org/W9"})

    assert work.id == "W9"
    assert work.abstract is None
    assert work.topic_id is None
    assert work.referenced_works == []
    assert work.cited_by_count == 0


def test_work_fetches_single_record():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        return httpx.Response(200, json=work_payload())

    assert client_for(handler).work("https://openalex.org/W1").id == "W1"
    assert seen["path"] == "/works/W1"


def test_pagination_follows_cursor_until_exhausted():
    pages = [
        {"results": [work_payload("W1")], "meta": {"next_cursor": "page-2"}},
        {"results": [work_payload("W2")], "meta": {"next_cursor": None}},
    ]
    cursors = []

    def handler(request):
        cursors.append(request.url.params["cursor"])
        return httpx.Response(200, json=pages[len(cursors) - 1])

    works = list(client_for(handler).works_by_topic("T11273"))

    assert [w.id for w in works] == ["W1", "W2"]
    assert cursors == ["*", "page-2"]


def test_pagination_stops_when_a_page_is_empty():
    def handler(request):
        return httpx.Response(200, json={"results": [], "meta": {"next_cursor": "never-ends"}})

    assert list(client_for(handler).citing_works("W1")) == []


def test_topic_query_applies_year_bounds():
    captured = {}

    def handler(request):
        captured["filter"] = request.url.params["filter"]
        return httpx.Response(200, json={"results": [], "meta": {}})

    list(client_for(handler).works_by_topic("T11273", from_year=2015, to_year=2020))

    assert captured["filter"] == (
        "primary_topic.id:T11273,from_publication_date:2015-01-01,to_publication_date:2020-12-31"
    )


def test_citing_works_uses_cites_filter():
    captured = {}

    def handler(request):
        captured["filter"] = request.url.params["filter"]
        return httpx.Response(200, json={"results": [], "meta": {}})

    list(client_for(handler).citing_works("https://openalex.org/W1"))

    assert captured["filter"] == "cites:W1"


def test_works_batches_ids_into_or_filters():
    ids = [f"W{n}" for n in range(MAX_IDS_PER_FILTER + 10)]
    batches = []

    def handler(request):
        batches.append(request.url.params["filter"].removeprefix("openalex_id:").split("|"))
        return httpx.Response(200, json={"results": [], "meta": {}})

    list(client_for(handler).works(ids))

    assert [len(batch) for batch in batches] == [MAX_IDS_PER_FILTER, 10]
    assert [i for batch in batches for i in batch] == ids


def test_referenced_works_expands_the_reference_list():
    def handler(request):
        if request.url.path == "/works/W1":
            return httpx.Response(200, json=work_payload("W1"))
        assert request.url.params["filter"] == "openalex_id:W2|W3"
        return httpx.Response(
            200, json={"results": [work_payload("W2"), work_payload("W3")], "meta": {}}
        )

    assert [w.id for w in client_for(handler).referenced_works("W1")] == ["W2", "W3"]


def test_retries_rate_limited_request_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    responses = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(429),
        httpx.Response(200, json=work_payload()),
    ]

    def handler(request):
        return responses.pop(0)

    assert client_for(handler, backoff=2.0).work("W1").id == "W1"
    assert slept == [7.0, 4.0]


def test_retry_budget_is_finite(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)

    def handler(request):
        return httpx.Response(503)

    with pytest.raises(OpenAlexError, match="gave up after 3 retries"):
        client_for(handler, max_retries=3).work("W1")


def test_client_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404)

    with pytest.raises(OpenAlexError, match="404"):
        client_for(handler).work("W-missing")

    assert len(calls) == 1


def test_contact_address_joins_the_polite_pool():
    captured = {}

    def handler(request):
        captured["mailto"] = request.url.params.get("mailto")
        captured["agent"] = request.headers["User-Agent"]
        return httpx.Response(200, json=work_payload())

    client_for(handler, mailto="someone@example.com").work("W1")

    assert captured["mailto"] == "someone@example.com"
    assert "someone@example.com" in captured["agent"]


def test_contact_address_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("OPENALEX_MAILTO", "env@example.com")
    captured = {}

    def handler(request):
        captured["mailto"] = request.url.params.get("mailto")
        return httpx.Response(200, json=work_payload())

    client_for(handler).work("W1")

    assert captured["mailto"] == "env@example.com"


def test_title_falls_back_to_display_name():
    payload = work_payload(title=None, display_name="Gated Graph Sequence Neural Networks")

    assert Work.model_validate(payload).title == "Gated Graph Sequence Neural Networks"
