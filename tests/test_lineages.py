import math

import pytest
from pydantic import ValidationError

from gnn.lineages import Lineage, _thinned, assess, judge, load_gold
from gnn.originators import flow, learned_flow
from test_originators import DIAMOND, FixedOpinion, lineage


def test_a_ranking_with_every_originator_on_top_scores_one():
    assert judge(["A", "B", "C"], [["A"], ["B"]], top=3) == (1.0, 1.0)


def test_a_missed_originator_costs_recall_and_a_late_one_costs_ndcg():
    recall, ndcg = judge(["X", "A", "Y"], [["A"], ["B"]], top=3)

    assert recall == 0.5
    assert ndcg == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_any_id_of_a_work_finds_it_but_only_once():
    assert judge(["A2", "A1"], [["A1", "A2"]], top=2) == (1.0, 1.0)


def test_more_originators_than_places_do_not_lower_the_ideal_ranking():
    _, ndcg = judge(["A", "B"], [["A"], ["B"], ["C"]], top=2)

    assert ndcg == 1.0


def test_a_paper_with_nothing_findable_has_nothing_to_score():
    assert all(math.isnan(value) for value in judge(["A"], [], top=1))


def test_gold_files_are_checked_on_load(tmp_path):
    path = tmp_path / "gold.yaml"
    path.write_text("lineages:\n- target: W1\n  split: dev\n  originators: [[W2, W3], [W4]]\n")
    expected = Lineage(target="W1", split="dev", originators=[["W2", "W3"], ["W4"]])
    assert load_gold(path) == [expected]

    path.write_text("lineages:\n  - target: W1\n    split: train\n    originators: [[W2]]\n")
    with pytest.raises(ValidationError):
        load_gold(path)

    with pytest.raises(FileNotFoundError):
        load_gold(tmp_path / "absent.yaml")


def test_thinning_drops_citations_but_keeps_every_work():
    graph = lineage([(f"P{n}", f"P{n + 1}") for n in range(200)])

    thinned = _thinned(graph, 0.5, seed=0)

    assert set(thinned) == set(graph)
    assert 60 < thinned.number_of_edges() < 140
    assert thinned.nodes["P0"] == graph.nodes["P0"]


def test_the_unweighted_walk_is_the_learned_walk_of_an_indifferent_model():
    graph = lineage(DIAMOND)

    indifferent = learned_flow(graph, "T", FixedOpinion(graph, graph.edges))

    assert flow(graph, "T") == pytest.approx(indifferent)


def test_an_assessment_judges_covers_and_stresses_every_scorer():
    graph = lineage([("T", "A"), ("T", "B"), ("A", "G"), ("B", "X"), ("U", "T")])
    gold = [
        Lineage(target="T", split="dev", originators=[["G"], ["NEVER-CRAWLED"]]),
        Lineage(target="B", split="test", originators=[["X"]]),
        Lineage(target="ABSENT", split="test", originators=[["G"]]),
    ]

    result = assess(graph, gold, FixedOpinion(graph, [("T", "A"), ("A", "G")]), seeds=1)
    dev, test = result["splits"]["dev"], result["splits"]["test"]

    assert result["targets_in_graph"] == "2/3"
    assert (dev["coverage"], test["coverage"]) == ("1/2", "1/1")
    assert set(dev["scorers"]) >= {"learned", "uniform_flow", "in_degree", "gateway"}
    assert dev["scorers"]["learned"]["recall"] == 1.0
    assert dev["scorers"]["learned"]["found"] == "1/2"
    assert "U" not in result["targets"][0]["scorers"]["learned"]["top"]
