import pytest

from gnn.evaluate import SAMPLERS, evaluate_cut, rolling_cuts, summarise
from gnn.train import TrainConfig
from test_dataset import cohorts
from test_train import preferential_attachment


@pytest.fixture(scope="module")
def rows():
    return evaluate_cut(
        preferential_attachment(), candidates=20, config=TrainConfig(epochs=80, patience=20)
    )


def scored(rows, sampler, scorer):
    return next(row for row in rows if row["sampler"] == sampler and row["scorer"] == scorer)


def test_every_scorer_meets_every_sampler(rows):
    assert {(row["sampler"], row["scorer"]) for row in rows} == {
        (sampler, scorer)
        for sampler in SAMPLERS
        for scorer in ("model", "recency", "in_degree", "logistic")
    }


def test_recency_carries_no_signal_once_negatives_match_the_cited_year(rows):
    recency = scored(rows, "year_matched", "recency")

    assert recency["roc_auc"] == pytest.approx(0.5)
    assert recency["mrr"] == pytest.approx(1 / 11)
    assert recency["hits_at_10"] == 0


def test_popularity_survives_the_matched_sampler_on_a_graph_built_from_it(rows):
    assert scored(rows, "year_matched", "in_degree")["roc_auc"] > 0.55


def test_the_model_learns_what_the_graph_holds(rows):
    assert scored(rows, "uniform", "model")["roc_auc"] > 0.7


def test_each_cut_ends_earlier_and_keeps_undated_work():
    graph = cohorts()
    graph.add_node("WU", publication_year=None, authors=[])
    graph.add_edge("WU", "W0")

    views = rolling_cuts(graph, cuts=3, window=0.2)
    latest = [max(view.nodes[n]["publication_year"] or 0 for n in view) for view in views]

    assert views[0].number_of_nodes() == graph.number_of_nodes()
    assert latest == [2008, 2007, 2006]
    assert all("WU" in view for view in views)


def test_the_summary_spreads_each_metric_across_runs():
    rows = [
        {"sampler": "uniform", "scorer": "model", "roc_auc": auc, "mrr": 0.5, "hits_at_10": 1.0}
        for auc in (0.6, 0.8)
    ]

    summary = summarise(rows)["uniform"]["model"]

    assert summary["roc_auc"] == {"mean": pytest.approx(0.7), "std": pytest.approx(0.1)}
    assert summary["mrr"]["std"] == 0
