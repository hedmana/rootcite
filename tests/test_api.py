from pathlib import Path

import networkx as nx
import pytest
import torch
from fastapi.testclient import TestClient

from api.app import create_app
from api.state import Field, Library
from graph.config import load_field
from llm.base import ProviderError
from llm.narrative import Claim, Judgement, Narrative, Verdict

CONFIG = """
name: gnn
display_name: Graph Neural Networks
description: >
  Architectures that operate
  directly on graphs.
topic_id: null
seed_papers: []
crawl:
  hop_depth: 2
  date_range: { start: null, end: null }
"""


class Indifferent:
    """A decoder with no favourites, so the walk splits evenly and stays predictable."""

    def eval(self):
        return self

    def decode(self, z, edge_label_index):
        return torch.zeros(edge_label_index.size(1))


class Scripted:
    """A model that says what it was told to, or fails the way a backend fails."""

    name = "scripted"

    def __init__(self, draft=None, verdicts=None, failure=None):
        self.draft = draft
        self.verdicts = verdicts
        self.failure = failure
        self.calls = 0

    def structured(self, prompt, schema, *, system=None):
        self.calls += 1
        if self.failure:
            raise self.failure
        return self.verdicts if schema is Judgement else self.draft


class Shelf(Library):
    """A library whose snapshot is in memory, counting how often it is built."""

    def __init__(self, fields_dir: Path, graph: nx.DiGraph, ready: bool = True):
        super().__init__(data_dir=fields_dir / "absent", fields_dir=fields_dir)
        self.graph = graph
        self.ready = ready
        self.builds = 0

    def snapshot(self, name):
        return Path("in-memory") if self.ready else None

    def _build(self, name):
        self.builds += 1
        position = {node: order for order, node in enumerate(self.graph)}
        return Field(
            load_field(name, self.fields_dir),
            self.graph,
            Indifferent(),
            (position, torch.zeros(len(position), 2)),
        )


def lineage() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("W1", title="The target", publication_year=2018, abstract="We build on convs.")
    for work, year in (("W2", 2016), ("W3", 2013)):
        graph.add_node(
            work, title=f"Paper {work}", publication_year=year, abstract=f"{work} did the thing."
        )
    graph.add_edges_from([("W1", "W2"), ("W2", "W3")])
    return graph


GROUNDED = Narrative(
    summary="The lineage runs through spectral methods.",
    claims=[Claim(work_id="W2", contribution="gave the spectral basis")],
)
SUPPORTED = Judgement(verdicts=[Verdict(claim=0, supported=True, reason="the abstract says so")])
REFUSED = Judgement(verdicts=[Verdict(claim=0, supported=False, reason="the abstract is silent")])


@pytest.fixture
def fields_dir(tmp_path):
    directory = tmp_path / "fields"
    directory.mkdir()
    (directory / "gnn.yaml").write_text(CONFIG)
    return directory


def client(fields_dir, graph=None, provider=None, ready=True):
    shelf = Shelf(fields_dir, graph or lineage(), ready=ready)
    return TestClient(create_app(shelf, provider, origins=["http://localhost:5173"])), shelf


def test_health_reports_which_fields_can_be_answered_about(fields_dir):
    ready, _ = client(fields_dir)
    cold, _ = client(fields_dir, ready=False)

    assert ready.get("/health").json() == {"status": "ok", "fields": {"gnn": True}}
    assert cold.get("/health").json()["fields"] == {"gnn": False}


def test_fields_describes_the_config_without_exposing_the_snapshot(fields_dir):
    api, _ = client(fields_dir)

    [field] = api.get("/fields").json()

    assert field == {
        "name": "gnn",
        "display_name": "Graph Neural Networks",
        "description": "Architectures that operate directly on graphs.",
        "ready": True,
    }


def test_lineage_ranks_the_ancestors_and_names_them(fields_dir):
    api, _ = client(fields_dir)

    body = api.get("/lineage/gnn/W1").json()

    assert body["target"]["title"] == "The target"
    # W3 carries more of the walk, and loses the lead to five more years of decay.
    assert [work["work_id"] for work in body["originators"]] == ["W2", "W3"]
    assert body["originators"][0]["score"] > 0
    assert set(body["baselines"]) == {
        "in_degree",
        "gateway",
        "path_weight",
        "time_decayed_pagerank",
    }


def test_the_snapshot_is_built_once_and_reused(fields_dir):
    api, shelf = client(fields_dir)

    api.get("/lineage/gnn/W1")
    api.get("/lineage/gnn/W2")

    assert shelf.builds == 1


def test_a_work_id_that_is_not_an_openalex_id_never_reaches_the_graph(fields_dir):
    api, shelf = client(fields_dir)

    assert api.get("/lineage/gnn/../../etc/passwd").status_code == 404
    assert api.get("/lineage/gnn/notanid").status_code == 422
    assert api.post("/narrative/gnn/notanid").status_code == 422
    assert shelf.builds == 0


def test_an_unknown_field_is_not_looked_for_on_disk(fields_dir):
    api, shelf = client(fields_dir)

    assert api.get("/lineage/nosuchfield/W1").status_code == 404
    assert shelf.builds == 0


def test_a_field_with_no_snapshot_says_so_rather_than_failing(fields_dir):
    api, _ = client(fields_dir, ready=False)

    response = api.get("/lineage/gnn/W1")

    assert response.status_code == 409
    assert "crawl and build" in response.json()["detail"]


def test_a_work_outside_the_snapshot_is_a_404(fields_dir):
    api, _ = client(fields_dir)

    assert api.get("/lineage/gnn/W999").status_code == 404


def test_the_top_count_is_bounded(fields_dir):
    api, _ = client(fields_dir)

    assert api.get("/lineage/gnn/W1", params={"top": 0}).status_code == 422
    assert api.get("/lineage/gnn/W1", params={"top": 500}).status_code == 422


def test_narrative_returns_the_claims_that_survived_verification(fields_dir):
    api, _ = client(fields_dir, provider=Scripted(GROUNDED, SUPPORTED))

    body = api.post("/narrative/gnn/W1").json()

    assert body["summary"] == GROUNDED.summary
    assert body["claims"] == [
        {"work_id": "W2", "title": "Paper W2", "contribution": "gave the spectral basis"}
    ]
    assert body["dropped"] == []
    assert body["partial"] is False


def test_a_dropped_claim_reaches_the_client_with_its_reason(fields_dir):
    api, _ = client(fields_dir, provider=Scripted(GROUNDED, REFUSED))

    body = api.post("/narrative/gnn/W1", params={"attempts": 1}).json()

    assert body["claims"] == []
    assert body["dropped"] == [{"work_id": "W2", "reason": "the abstract is silent"}]
    assert body["partial"] is True


def test_a_backend_failure_is_a_502_that_keeps_the_backends_words_to_itself(fields_dir):
    failure = ProviderError("401 from https://api.example: key sk-secret rejected")
    api, _ = client(fields_dir, provider=Scripted(failure=failure))

    response = api.post("/narrative/gnn/W1")

    assert response.status_code == 502
    assert "sk-secret" not in response.text
    assert "api.example" not in response.text


def test_a_paper_with_no_ancestors_costs_nothing(fields_dir):
    graph = lineage()
    provider = Scripted(GROUNDED, SUPPORTED)
    api, _ = client(fields_dir, graph=graph, provider=provider)

    assert api.post("/narrative/gnn/W3").status_code == 404
    assert provider.calls == 0


def test_only_the_allowed_origin_is_answered(fields_dir):
    api, _ = client(fields_dir)

    allowed = api.get("/health", headers={"Origin": "http://localhost:5173"})
    other = api.get("/health", headers={"Origin": "https://elsewhere.example"})

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in other.headers
