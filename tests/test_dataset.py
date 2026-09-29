import networkx as nx
import pytest
import torch

from gnn.dataset import (
    FEATURE_NAMES,
    build_dataset,
    index_graph,
    load_dataset,
    sample_negatives,
    save_dataset,
)


def citation_graph(years, edges):
    graph = nx.DiGraph()
    for node, year in years.items():
        graph.add_node(
            node,
            title=f"Paper {node}",
            abstract="an abstract",
            publication_year=year,
            cited_by_count=3,
            authors=["Ada Lovelace"],
        )
    graph.add_edges_from(edges)
    return graph


def cohorts(spans=8, per_year=3, first_year=2001):
    """Successive cohorts, each citing two papers from the cohort before it."""
    years = {f"W{i}": first_year + i // per_year for i in range(spans * per_year)}
    edges = [
        (f"W{i}", f"W{j}")
        for i in range(per_year, len(years))
        for j in (i - per_year, i - per_year - 1)
        if j >= 0
    ]
    return citation_graph(years, edges)


def positive_source_years(graph, dataset, split):
    data = getattr(dataset, split)
    sources = data.edge_label_index[0][data.edge_label == 1]
    return {
        graph.nodes[dataset.node_ids[source]]["publication_year"] for source in sources.tolist()
    }


def test_splits_are_cuts_in_time_not_random_samples():
    graph = cohorts()
    dataset = build_dataset(graph, val_fraction=0.2, test_fraction=0.2)

    train = positive_source_years(graph, dataset, "train")
    val = positive_source_years(graph, dataset, "val")
    test = positive_source_years(graph, dataset, "test")

    assert max(train) < min(val)
    assert max(val) < min(test)
    assert dataset.report.train_edges > dataset.report.val_edges


def test_evaluation_splits_see_only_the_edges_that_preceded_them():
    dataset = build_dataset(cohorts(), val_fraction=0.2, test_fraction=0.2)

    assert dataset.val.edge_index.equal(dataset.train.edge_index)
    assert dataset.test.edge_index.size(1) == (
        dataset.train.edge_index.size(1) + dataset.report.val_edges
    )


def test_degree_features_come_from_the_training_subgraph_alone():
    dataset = build_dataset(cohorts(), val_fraction=0.2, test_fraction=0.2)
    newest = dataset.node_ids.index("W23")

    in_degree = FEATURE_NAMES.index("log_in_degree")
    out_degree = FEATURE_NAMES.index("log_out_degree")

    assert dataset.train.x[newest, out_degree] == 0
    assert dataset.train.x[newest, in_degree] == 0


def test_negatives_are_citations_that_could_have_happened_and_did_not():
    graph = cohorts()
    dataset = build_dataset(graph, val_fraction=0.2, test_fraction=0.2)
    years = {node: graph.nodes[node]["publication_year"] for node in dataset.node_ids}

    for split in ("train", "val", "test"):
        data = getattr(dataset, split)
        sampled = data.edge_label_index[:, data.edge_label == 0]
        for source, target in sampled.t().tolist():
            citing, cited = dataset.node_ids[source], dataset.node_ids[target]
            assert citing != cited
            assert years[cited] <= years[citing]
            assert not graph.has_edge(citing, cited)


def test_year_matched_negatives_share_the_cited_works_year():
    graph = cohorts()
    node_ids, years, edges = index_graph(graph)
    count = len(node_ids)
    existing = (edges[0] * count + edges[1]).sort().values
    positive = edges.repeat_interleave(5, dim=1)

    negative, owner = sample_negatives(
        positive, years, existing, count, torch.Generator().manual_seed(0), year_matched=True
    )

    assert negative.size(1) > 0
    assert negative[0].equal(positive[0, owner])
    assert years[negative[1]].equal(years[positive[1, owner]])
    assert not torch.isin(negative[0] * count + negative[1], existing).any()
    assert (negative[0] != negative[1]).all()


def test_negatives_balance_the_positives_where_the_graph_leaves_room():
    dataset = build_dataset(cohorts(), val_fraction=0.2, test_fraction=0.2)

    assert dataset.report.negatives_sampled == dataset.report.negatives_wanted
    for split in ("train", "val", "test"):
        label = getattr(dataset, split).edge_label
        assert (label == 1).sum() == (label == 0).sum()


def test_a_saturated_source_yields_fewer_negatives_than_positives():
    """A paper that already cites everything it could has no negative to offer."""
    graph = citation_graph({"A": 2001, "B": 2002}, [("B", "A")])

    dataset = build_dataset(graph, val_fraction=0.0, test_fraction=0.0)

    assert dataset.report.negatives_sampled == 0
    assert dataset.report.negatives_wanted == 1


def test_the_seed_fixes_the_negatives():
    graph = cohorts()

    first = build_dataset(graph, seed=3).train.edge_label_index
    again = build_dataset(graph, seed=3).train.edge_label_index
    other = build_dataset(graph, seed=4).train.edge_label_index

    assert first.equal(again)
    assert not first.equal(other)


def test_undated_citations_carry_structure_but_never_supervise():
    graph = cohorts()
    graph.add_node("WU", publication_year=None, cited_by_count=0, authors=[])
    graph.add_edge("WU", "W0")

    dataset = build_dataset(graph, val_fraction=0.2, test_fraction=0.2)
    undated = dataset.node_ids.index("WU")

    assert dataset.report.undated_edges == 1
    assert undated in dataset.train.edge_index[0].tolist()
    for split in ("train", "val", "test"):
        assert undated not in getattr(dataset, split).edge_label_index[0].tolist()


def test_a_graph_without_dates_yields_no_supervision():
    graph = citation_graph({"A": None, "B": None}, [("A", "B")])

    dataset = build_dataset(graph)

    assert dataset.report.train_edges == 0
    assert dataset.train.edge_label.numel() == 0
    assert dataset.train.edge_index.size(1) == 1


def test_snapshot_round_trip_preserves_every_split(tmp_path):
    dataset = build_dataset(cohorts(), val_fraction=0.2, test_fraction=0.2, seed=5)

    reloaded = load_dataset(save_dataset(tmp_path, dataset))

    assert reloaded.node_ids == dataset.node_ids
    assert reloaded.report == dataset.report
    assert reloaded.train.x.equal(dataset.train.x)
    for split in ("train", "val", "test"):
        original, copy = getattr(dataset, split), getattr(reloaded, split)
        assert copy.edge_index.equal(original.edge_index)
        assert copy.edge_label_index.equal(original.edge_label_index)
        assert copy.edge_label.equal(original.edge_label)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_every_split_carries_the_declared_feature_matrix(split):
    dataset = build_dataset(cohorts(), val_fraction=0.2, test_fraction=0.2)

    assert getattr(dataset, split).x.shape == (len(dataset.node_ids), len(FEATURE_NAMES))
