import json

import numpy as np
import pandas as pd
import pytest

from graph.build import (
    clean,
    latest_snapshot,
    load_snapshot,
    save_snapshot,
    snapshot_dir,
    summarize,
    to_graph,
)
from graph.config import DateRange


def node(work_id, year=2015, abstract="an abstract", title=None):
    return {
        "id": work_id,
        "title": title if title is not None else f"Paper {work_id}",
        "abstract": abstract,
        "publication_year": year,
        "publication_date": f"{year}-01-01",
        "cited_by_count": 10,
        "topic_id": "T1",
        "authors": ["Ada Lovelace"],
    }


def frames(nodes, edges):
    return pd.DataFrame(nodes), pd.DataFrame(edges, columns=["source", "target"])


def test_duplicate_nodes_and_edges_are_collapsed():
    nodes, edges = frames([node("A"), node("A"), node("B")], [("A", "B"), ("A", "B")])

    nodes, edges, report = clean(nodes, edges)

    assert report.duplicate_nodes == 1
    assert report.duplicate_edges == 1
    assert report.nodes == 2
    assert report.edges == 1


def test_self_loops_are_removed():
    nodes, edges = frames([node("A"), node("B")], [("A", "A"), ("A", "B")])

    _, edges, report = clean(nodes, edges)

    assert report.self_loops == 1
    assert list(edges.itertuples(index=False, name=None)) == [("A", "B")]


def test_edges_past_the_crawl_horizon_are_dropped():
    nodes, edges = frames([node("A"), node("B")], [("A", "B"), ("B", "OUTSIDE")])

    _, edges, report = clean(nodes, edges)

    assert report.dangling_edges == 1
    assert list(edges["target"]) == ["B"]


def test_nodes_outside_the_date_range_are_dropped_with_their_edges():
    nodes, edges = frames(
        [node("A", year=2015), node("B", year=1970), node("C", year=2016)],
        [("A", "B"), ("A", "C")],
    )

    nodes, edges, report = clean(nodes, edges, date_range=DateRange(start=2000, end=2020))

    assert report.out_of_range_nodes == 1
    assert sorted(nodes["id"]) == ["A", "C"]
    assert list(edges.itertuples(index=False, name=None)) == [("A", "C")]


def test_only_the_largest_component_survives_by_default():
    nodes, edges = frames(
        [node("A"), node("B"), node("C"), node("X"), node("Y")],
        [("A", "B"), ("B", "C"), ("X", "Y")],
    )

    nodes, _, report = clean(nodes, edges)

    assert report.disconnected_nodes == 2
    assert sorted(nodes["id"]) == ["A", "B", "C"]


def test_every_component_can_be_kept():
    nodes, edges = frames([node("A"), node("B"), node("X"), node("Y")], [("A", "B"), ("X", "Y")])

    nodes, _, report = clean(nodes, edges, largest_component_only=False)

    assert report.disconnected_nodes == 0
    assert len(nodes) == 4


def test_cleaning_an_empty_crawl_is_not_an_error():
    nodes, edges = frames([], [])
    nodes["id"] = nodes.get("id", pd.Series(dtype=str))

    nodes, edges, report = clean(nodes, edges)

    assert report.nodes == 0
    assert report.edges == 0


def test_graph_edges_point_from_citing_to_cited_work():
    nodes, edges = frames([node("A"), node("B")], [("A", "B")])

    graph = to_graph(nodes, edges)

    assert graph.has_edge("A", "B")
    assert not graph.has_edge("B", "A")
    assert graph.nodes["A"]["title"] == "Paper A"
    assert list(graph.nodes["A"]["authors"]) == ["Ada Lovelace"]


def test_summary_counts_absent_abstracts_and_years():
    nodes, edges = frames(
        [
            node("A"),
            node("B", abstract=None),
            node("C", abstract=""),
            node("D", year=None),
        ],
        [("A", "B"), ("A", "C"), ("A", "D")],
    )
    nodes["abstract"] = nodes["abstract"].astype(object).where(nodes["abstract"].notna(), np.nan)

    stats = summarize(to_graph(nodes, edges))

    assert stats.nodes_missing_abstract == 2
    assert stats.nodes_missing_year == 1
    assert stats.year_min == 2015


def test_summary_ranks_the_most_cited_works_inside_the_graph():
    nodes, edges = frames(
        [node("A"), node("B"), node("ROOT")], [("A", "ROOT"), ("B", "ROOT"), ("A", "B")]
    )

    stats = summarize(to_graph(nodes, edges), top=2)

    assert stats.most_cited_within_graph[0] == ("ROOT", "Paper ROOT", 2)
    assert len(stats.most_cited_within_graph) == 2


def test_summary_of_an_empty_graph_does_not_divide_by_zero():
    stats = summarize(to_graph(*frames([], [])))

    assert stats.nodes == 0
    assert stats.mean_in_degree == 0
    assert stats.year_min is None


def test_snapshot_directory_is_named_by_field_and_date(tmp_path):
    from datetime import date

    path = snapshot_dir(tmp_path, "gnn", on=date(2026, 9, 2))

    assert path == tmp_path / "gnn" / "snapshots" / "gnn-20260902"


def test_snapshot_round_trips_through_disk(tmp_path):
    from datetime import date

    nodes, edges = frames([node("A"), node("B")], [("A", "B")])
    nodes, edges, report = clean(nodes, edges)
    stats = summarize(to_graph(nodes, edges))
    path = snapshot_dir(tmp_path, "gnn", on=date(2026, 9, 2))

    save_snapshot(path, nodes, edges, stats, report)
    graph = load_snapshot(path)

    assert graph.has_edge("A", "B")
    assert graph.nodes["B"]["title"] == "Paper B"
    assert (path / "stats.json").exists()


def test_latest_snapshot_picks_the_most_recent_date(tmp_path):
    for stamp in ("gnn-20260101", "gnn-20260902", "gnn-20250601"):
        (tmp_path / "gnn" / "snapshots" / stamp).mkdir(parents=True)

    assert latest_snapshot(tmp_path, "gnn").name == "gnn-20260902"


def test_latest_snapshot_reports_when_there_is_none(tmp_path):
    with pytest.raises(FileNotFoundError, match="no snapshot"):
        latest_snapshot(tmp_path, "gnn")


def test_seeds_lost_to_cleaning_are_reported(tmp_path, monkeypatch):
    from graph import build

    nodes, edges = frames([node("A"), node("B"), node("LONELY")], [("A", "B")])
    raw = tmp_path / "gnn" / "raw"
    raw.mkdir(parents=True)
    nodes.to_parquet(raw / "nodes.parquet", index=False)
    edges.to_parquet(raw / "edges.parquet", index=False)

    config = build.load_field("gnn")
    monkeypatch.setattr(config, "seed_papers", ["A", "LONELY"])
    monkeypatch.setattr(build, "load_field", lambda _: config)

    path, _ = build.build_field("gnn", data_dir=tmp_path)
    cleaning = json.loads((path / "stats.json").read_text())["cleaning"]

    assert cleaning["dropped_seeds"] == ["LONELY"]
