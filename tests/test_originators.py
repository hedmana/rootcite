import math

import networkx as nx
import pandas as pd
import pytest
import torch

from gnn.content import Content
from gnn.dataset import build_dataset
from gnn.originators import embed, learned_flow, score_target
from gnn.train import TrainConfig, save_model, train
from graph.build import CleaningReport, save_snapshot, snapshot_dir, summarize, to_graph
from graph.score import rank


class FixedOpinion:
    """A model with settled views: the favoured citations are plausible, the rest are not."""

    def __init__(self, graph, favoured):
        position = {node: order for order, node in enumerate(graph.nodes)}
        self.favoured = {(position[source], position[target]) for source, target in favoured}

    def eval(self):
        return self

    def encode(self, x, edge_index):
        return torch.zeros(x.size(0), 2)

    def decode(self, z, edge_label_index):
        return torch.tensor(
            [
                4.0 if (source, target) in self.favoured else -4.0
                for source, target in edge_label_index.t().tolist()
            ]
        )


def lineage(edges, years=None):
    graph = nx.DiGraph(edges)
    for node in graph:
        graph.nodes[node]["publication_year"] = (years or {}).get(node, 2015)
        graph.nodes[node]["authors"] = ["Ada Lovelace"]
        graph.nodes[node]["title"] = f"Paper {node}"
    return graph


DIAMOND = [("T", "A"), ("T", "B"), ("A", "G"), ("B", "G")]


def test_an_indifferent_model_splits_the_walk_evenly():
    graph = lineage(DIAMOND)

    scores = learned_flow(graph, "T", FixedOpinion(graph, graph.edges), damping=0.85)

    assert scores["A"] == pytest.approx(0.425)
    assert scores["B"] == pytest.approx(0.425)
    assert scores["G"] == pytest.approx(0.7225)


def test_the_walk_follows_the_citations_the_model_believes_in():
    graph = lineage([("T", "A"), ("T", "B"), ("A", "X"), ("B", "Y")])

    scores = learned_flow(graph, "T", FixedOpinion(graph, [("T", "A"), ("A", "X"), ("B", "Y")]))

    assert scores["A"] > scores["B"]
    assert scores["X"] > scores["Y"]


def test_structural_twins_are_separated_by_the_model_alone():
    """A and B are indistinguishable in the graph, so only the model can rank them."""
    graph = lineage(DIAMOND)

    scores = learned_flow(graph, "T", FixedOpinion(graph, [("T", "A")]))

    assert scores["A"] > scores["B"]


def test_structural_twins_are_separated_by_what_they_are_about():
    """A precursor and a tool the model finds equally plausible: content tells them apart."""
    graph = lineage(DIAMOND)
    abstracts = {"T": "graph attention", "A": "attention alignment", "B": "stochastic optimiser"}
    for node, abstract in abstracts.items():
        graph.nodes[node]["abstract"] = abstract
    indifferent = FixedOpinion(graph, graph.edges)

    plain = learned_flow(graph, "T", indifferent)
    informed = learned_flow(graph, "T", indifferent, content=Content(graph))

    assert plain["A"] == plain["B"]
    assert informed["A"] > informed["B"]


def test_the_distant_past_is_discounted():
    graph = lineage([("T", "RECENT"), ("T", "ANCIENT")], years={"T": 2020, "RECENT": 2018})
    graph.nodes["ANCIENT"]["publication_year"] = 1970

    scores = learned_flow(graph, "T", FixedOpinion(graph, graph.edges), half_life=10)

    assert scores["RECENT"] > scores["ANCIENT"]


def test_a_paper_is_not_its_own_originator():
    graph = lineage(DIAMOND)

    assert "T" not in learned_flow(graph, "T", FixedOpinion(graph, graph.edges))


def test_no_ancestor_outweighs_the_paper_it_is_traced_from():
    graph = lineage(DIAMOND)

    scores = learned_flow(graph, "T", FixedOpinion(graph, graph.edges), damping=0.99)

    assert all(0 < score <= 1 for score in scores.values())


def test_an_embedding_can_be_computed_once_and_reused():
    graph = lineage(DIAMOND)
    model = FixedOpinion(graph, graph.edges)

    shared = embed(graph, model)

    assert learned_flow(graph, "T", model, embedding=shared) == learned_flow(graph, "T", model)


def deep_lineage(width=4, depth=6):
    edges = [
        (f"L{level}_{index}", f"L{level + 1}_{(index + step) % width}")
        for level in range(depth)
        for index in range(width)
        for step in (0, 1)
    ]
    years = {
        f"L{level}_{index}": 2020 - level for level in range(depth + 1) for index in range(width)
    }
    graph = lineage(edges, years)
    graph.add_edges_from(("ROOT", f"L0_{index}") for index in range(width))
    graph.nodes["ROOT"]["publication_year"] = 2021
    graph.nodes["ROOT"]["authors"] = ["Ada Lovelace"]
    return graph


def test_a_trained_model_scores_every_ancestor_it_can_reach():
    graph = deep_lineage()
    model, _ = train(build_dataset(graph, seed=1), config=TrainConfig(epochs=10, patience=10))

    scores = learned_flow(graph, "ROOT", model)

    assert set(scores) == set(graph) - {"ROOT"}
    assert all(score > 0 and math.isfinite(score) for score in scores.values())
    assert rank(scores, top=3)


def test_scoring_a_field_reads_the_snapshot_and_the_model_beside_it(tmp_path):
    graph = deep_lineage()
    nodes = pd.DataFrame(
        {"id": node, **{key: value for key, value in data.items()}}
        for node, data in graph.nodes(data=True)
    )
    edges = pd.DataFrame(graph.edges(), columns=["source", "target"])
    path = snapshot_dir(tmp_path, "control")
    save_snapshot(path, nodes, edges, summarize(to_graph(nodes, edges)), CleaningReport())
    model, report = train(build_dataset(graph, seed=1), config=TrainConfig(epochs=10, patience=10))
    save_model(path, model, report)

    learned, baselines = score_target("control", "ROOT", data_dir=tmp_path)

    assert learned
    assert set(baselines) == {"in_degree", "gateway", "path_weight", "time_decayed_pagerank"}
