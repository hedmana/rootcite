import networkx as nx
import pytest

from graph.score import (
    SCORERS,
    acyclic,
    ancestry,
    compare,
    gateway,
    in_degree,
    overlap,
    path_weight,
    rank,
    time_decayed_pagerank,
)


def graph_from(edges, years=None):
    graph = nx.DiGraph(edges)
    for node in graph:
        graph.nodes[node]["publication_year"] = (years or {}).get(node, 2015)
        graph.nodes[node]["title"] = f"Paper {node}"
    return graph


# T cites A and B, both of which reach G, and only G reaches X and Y.
BOTTLENECK = [("T", "A"), ("T", "B"), ("A", "G"), ("B", "G"), ("G", "X"), ("G", "Y")]


def test_ancestry_keeps_only_what_the_target_reaches():
    graph = graph_from([*BOTTLENECK, ("UNRELATED", "T")])

    subgraph = ancestry(graph, "T")

    assert set(subgraph) == {"T", "A", "B", "G", "X", "Y"}


def test_ancestry_rejects_a_target_outside_the_graph():
    with pytest.raises(KeyError):
        ancestry(graph_from(BOTTLENECK), "MISSING")


def test_acyclic_drops_the_edge_that_disagrees_with_publication_order():
    graph = graph_from([("A", "B"), ("B", "A")], years={"A": 2020, "B": 2010})

    broken = acyclic(graph)

    assert list(broken.edges()) == [("A", "B")]


def test_acyclic_leaves_a_dag_alone():
    graph = graph_from(BOTTLENECK)

    assert acyclic(graph) is graph


def test_in_degree_counts_citations_from_within_the_lineage_only():
    graph = graph_from([*BOTTLENECK, ("UNRELATED", "G")])

    scores = in_degree(graph, "T")

    assert scores["G"] == 2
    assert scores["A"] == 1


def test_gateway_scores_a_bottleneck_by_what_it_alone_reaches():
    scores = gateway(graph_from(BOTTLENECK), "T")

    assert scores["G"] == 2
    assert scores["A"] == 0
    assert scores["B"] == 0


def test_gateway_excludes_the_target_it_is_rooted_at():
    assert "T" not in gateway(graph_from(BOTTLENECK), "T")


def test_path_weight_pays_for_reach_and_charges_for_distance():
    scores = path_weight(graph_from(BOTTLENECK), "T", decay=0.5)

    # G sits a hop further back than A, but is reached two ways rather than one.
    assert scores["G"] == pytest.approx(scores["A"])
    assert scores["X"] < scores["G"]


def test_time_decayed_pagerank_discounts_the_distant_past():
    edges = [("T", "RECENT"), ("T", "ANCIENT")]
    years = {"T": 2020, "RECENT": 2018, "ANCIENT": 1970}

    scores = time_decayed_pagerank(graph_from(edges, years), "T", half_life=10)

    assert scores["RECENT"] > scores["ANCIENT"]


def test_scorers_survive_nodes_with_no_publication_year():
    graph = graph_from(BOTTLENECK)
    del graph.nodes["G"]["publication_year"]

    for scorer in SCORERS.values():
        assert scorer(graph, "T")


def test_rank_returns_the_best_first_and_breaks_ties_by_id():
    ranked = rank({"B": 1.0, "A": 1.0, "C": 5.0}, top=2)

    assert ranked == [("C", 5.0), ("A", 1.0)]


def test_overlap_is_total_agreement_and_none_at_the_extremes():
    scores = {"A": 3.0, "B": 2.0}

    assert overlap(scores, scores, top=2) == 1.0
    assert overlap(scores, {"X": 3.0, "Y": 2.0}, top=2) == 0.0
    assert overlap(scores, {}, top=2) == 0.0


def test_compare_scores_every_baseline_on_one_lineage():
    scored = compare(graph_from(BOTTLENECK), "T")

    assert set(scored) == set(SCORERS)
    assert all(scores for scores in scored.values())
