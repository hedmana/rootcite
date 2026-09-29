import json

import httpx
import pandas as pd
import pytest

from graph import semanticscholar
from graph.build import RECOVERED_EDGES, load_raw
from graph.semanticscholar import (
    Reference,
    SemanticScholarClient,
    SemanticScholarError,
    mag_id,
    recover_edges,
)


def client_for(handler, **kwargs):
    return SemanticScholarClient(transport=httpx.MockTransport(handler), backoff=0, **kwargs)


def reference(title, year=2015, mag=None):
    return Reference(title=title, year=year, externalIds={"MAG": mag} if mag else None)


def nodes(*rows):
    return pd.DataFrame(rows, columns=["id", "title", "publication_year"])


def test_only_ids_made_from_mag_ids_are_asked_about():
    assert mag_id("W2519887557") == "2519887557"
    assert mag_id("W4297733535") is None
    assert mag_id("not-an-id") is None


def test_references_are_read_by_mag_id_and_unknown_works_skipped():
    asked = {}

    def handler(request):
        asked["ids"] = json.loads(request.content)["ids"]
        asked["fields"] = request.url.params["fields"]
        chebnet = {"references": [{"title": "ChebNet", "year": 2016}]}
        return httpx.Response(200, json=[chebnet, None])

    found = list(client_for(handler).references(["W1", "W2"]))

    assert asked["ids"] == ["MAG:1", "MAG:2"]
    assert "references.externalIds" in asked["fields"]
    assert found == [("W1", [Reference(title="ChebNet", year=2016)])]


def test_a_rate_limited_batch_is_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429) if len(calls) == 1 else httpx.Response(200, json=[None])

    assert list(client_for(handler).references(["W1"])) == []
    assert len(calls) == 2


def test_a_refused_batch_raises_rather_than_passing_for_empty():
    with pytest.raises(SemanticScholarError):
        list(client_for(lambda request: httpx.Response(403)).references(["W1"]))


def test_the_api_key_is_sent_only_when_one_is_set(monkeypatch):
    headers = []

    def handler(request):
        headers.append(request.headers.get("x-api-key"))
        return httpx.Response(200, json=[None])

    monkeypatch.delenv("S2_API_KEY", raising=False)
    list(client_for(handler).references(["W1"]))
    monkeypatch.setenv("S2_API_KEY", "from-the-environment")
    list(client_for(handler).references(["W1"]))

    assert headers == [None, "from-the-environment"]


def test_a_reference_is_matched_by_mag_id_first_then_by_title_and_year():
    crawled = nodes(
        ("W1", "Citing Work", 2016),
        ("W2", "Kept By Id", 2014),
        ("W3", "Spectral Networks and Locally Connected Networks on Graphs", 2013),
    )
    cited = [
        reference("anything", mag="2"),
        reference("Spectral networks and locally-connected networks on graphs.", 2014, mag="9"),
    ]

    edges = recover_edges(crawled, [("W1", cited)])

    assert list(edges.itertuples(index=False, name=None)) == [("W1", "W2"), ("W1", "W3")]


def test_a_title_that_names_no_single_work_matches_nothing():
    crawled = nodes(
        ("W1", "Citing Work", 2016),
        ("W2", "Graph Kernels", 2008),
        ("W3", "Graph Kernels", 2008),
        ("W4", "Introduction", 1990),
    )
    cited = [reference("Graph Kernels", 2008), reference("Introduction", 2010)]

    assert recover_edges(crawled, [("W1", cited)]).empty


def test_a_work_is_never_recovered_as_citing_itself():
    crawled = nodes(("W1", "Self Aware", 2016))

    assert recover_edges(crawled, [("W1", [reference("Self Aware", 2016)])]).empty


def test_recovered_citations_join_the_crawl_when_the_graph_is_built(tmp_path):
    pd.DataFrame({"id": ["W1", "W2"]}).to_parquet(tmp_path / "nodes.parquet")
    pd.DataFrame({"source": ["W1"], "target": ["W2"]}).to_parquet(tmp_path / "edges.parquet")
    pd.DataFrame({"source": ["W2"], "target": ["W1"]}).to_parquet(tmp_path / RECOVERED_EDGES)

    _, edges = load_raw(tmp_path)

    assert list(edges.itertuples(index=False, name=None)) == [("W1", "W2"), ("W2", "W1")]


def test_a_field_recovers_only_what_openalex_did_not_list(tmp_path, monkeypatch):
    raw = tmp_path / "gnn" / "raw"
    raw.mkdir(parents=True)
    nodes(("W1", "Citing", 2016), ("W2", "Listed", 2014), ("W3", "Lost", 2013)).to_parquet(
        raw / "nodes.parquet"
    )
    pd.DataFrame({"source": ["W1"], "target": ["W2"]}).to_parquet(raw / "edges.parquet")

    def handler(request):
        cited = [{"title": "Listed", "year": 2014}, {"title": "Lost", "year": 2013}]
        return httpx.Response(200, json=[{"references": cited}, None, None])

    path, matched, new = semanticscholar.recover_field(
        "gnn", data_dir=tmp_path, client=client_for(handler)
    )

    assert (matched, new) == (2, 1)
    assert path == raw / RECOVERED_EDGES
