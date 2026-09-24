"""Turn a raw crawl into a canonical, cleaned citation graph.

The crawler is deliberately credulous: it journals every edge it sees, including
edges pointing past the last hop it fetched. Everything that makes the graph
trustworthy happens here, and every discard is counted so the cost of cleaning
is visible rather than silent.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from graph.config import DateRange, FieldConfig, load_field
from graph.crawler import DATA_DIR

logger = logging.getLogger(__name__)

NODE_ATTRIBUTES = (
    "title",
    "abstract",
    "publication_year",
    "publication_date",
    "cited_by_count",
    "topic_id",
    "authors",
)


@dataclass
class CleaningReport:
    """What the cleaning stage removed, and what survived."""

    raw_nodes: int = 0
    raw_edges: int = 0
    duplicate_nodes: int = 0
    duplicate_edges: int = 0
    out_of_range_nodes: int = 0
    dangling_edges: int = 0
    self_loops: int = 0
    disconnected_nodes: int = 0
    nodes: int = 0
    edges: int = 0
    dropped_seeds: list[str] = field(default_factory=list)

    def log(self) -> None:
        for name, value in asdict(self).items():
            if name != "dropped_seeds":
                logger.info("%-20s %s", name, value)
        if self.dropped_seeds:
            logger.warning(
                "%d seed papers did not survive cleaning: %s",
                len(self.dropped_seeds),
                ", ".join(self.dropped_seeds),
            )


@dataclass
class GraphStats:
    """Sanity numbers for a built graph."""

    nodes: int
    edges: int
    density: float
    weakly_connected_components: int
    largest_component_share: float
    mean_in_degree: float
    max_in_degree: int
    max_out_degree: int
    isolated_nodes: int
    year_min: int | None
    year_max: int | None
    nodes_missing_year: int
    nodes_missing_abstract: int
    most_cited_within_graph: list[tuple[str, str, int]] = field(default_factory=list)


def is_missing(value: object) -> bool:
    """Absent in the pandas sense: None, NaN, or empty. NaN is truthy, so `not x` lies."""
    return value is None or (isinstance(value, float) and pd.isna(value)) or value == ""


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the crawler's raw node and edge tables."""
    nodes = pd.read_parquet(raw_dir / "nodes.parquet")
    edges = pd.read_parquet(raw_dir / "edges.parquet")
    return nodes, edges


def clean(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    date_range: DateRange | None = None,
    largest_component_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Reduce raw crawl tables to a consistent node set and edge set."""
    report = CleaningReport(raw_nodes=len(nodes), raw_edges=len(edges))

    nodes = nodes.drop_duplicates(subset="id")
    report.duplicate_nodes = report.raw_nodes - len(nodes)

    if date_range is not None:
        in_range = nodes["publication_year"].map(date_range.contains)
        report.out_of_range_nodes = int((~in_range).sum())
        nodes = nodes[in_range]

    edges = edges.drop_duplicates()
    report.duplicate_edges = report.raw_edges - len(edges)

    not_self = edges["source"] != edges["target"]
    report.self_loops = int((~not_self).sum())
    edges = edges[not_self]

    known = set(nodes["id"])
    connected = edges["source"].isin(known) & edges["target"].isin(known)
    report.dangling_edges = int((~connected).sum())
    edges = edges[connected]

    if largest_component_only:
        keep = _largest_component(nodes["id"], edges)
        report.disconnected_nodes = len(nodes) - len(keep)
        nodes = nodes[nodes["id"].isin(keep)]
        edges = edges[edges["source"].isin(keep) & edges["target"].isin(keep)]

    report.nodes = len(nodes)
    report.edges = len(edges)
    return nodes, edges, report


def _largest_component(node_ids: pd.Series, edges: pd.DataFrame) -> set[str]:
    scaffold = nx.Graph()
    scaffold.add_nodes_from(node_ids)
    scaffold.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))
    if scaffold.number_of_nodes() == 0:
        return set()
    return max(nx.connected_components(scaffold), key=len)


def _plain(value: object) -> object:
    """Parquet hands list columns back as arrays, which have no truth value."""
    return value.tolist() if isinstance(value, np.ndarray) else value


def to_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Assemble a directed graph whose edges point from citing to cited work."""
    graph = nx.DiGraph()
    attributes = [column for column in NODE_ATTRIBUTES if column in nodes.columns]
    for row in nodes.itertuples(index=False):
        graph.add_node(row.id, **{name: _plain(getattr(row, name)) for name in attributes})
    graph.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))
    return graph


def summarize(graph: nx.DiGraph, top: int = 10) -> GraphStats:
    """Compute the sanity numbers worth eyeballing after a build."""
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    components = list(nx.weakly_connected_components(graph))
    years = [year for _, year in graph.nodes(data="publication_year") if not is_missing(year)]
    node_count = graph.number_of_nodes() or 1

    ranked = sorted(in_degrees.items(), key=lambda item: item[1], reverse=True)[:top]
    return GraphStats(
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        density=nx.density(graph),
        weakly_connected_components=len(components),
        largest_component_share=(max((len(c) for c in components), default=0) / node_count),
        mean_in_degree=sum(in_degrees.values()) / node_count,
        max_in_degree=max(in_degrees.values(), default=0),
        max_out_degree=max(out_degrees.values(), default=0),
        isolated_nodes=sum(1 for node in graph if not in_degrees[node] and not out_degrees[node]),
        year_min=int(min(years)) if years else None,
        year_max=int(max(years)) if years else None,
        nodes_missing_year=graph.number_of_nodes() - len(years),
        nodes_missing_abstract=sum(
            1 for _, abstract in graph.nodes(data="abstract") if is_missing(abstract)
        ),
        most_cited_within_graph=[
            (
                node,
                "" if is_missing(title := graph.nodes[node].get("title")) else str(title),
                degree,
            )
            for node, degree in ranked
        ],
    )


def snapshot_dir(data_dir: Path, field_name: str, on: date | None = None) -> Path:
    """`data/<field>/snapshots/<field>-YYYYMMDD`, one directory per build."""
    stamp = (on or date.today()).strftime("%Y%m%d")
    return data_dir / field_name / "snapshots" / f"{field_name}-{stamp}"


def latest_snapshot(data_dir: Path, field_name: str) -> Path:
    """Most recent snapshot for a field, by the date in its name."""
    snapshots = sorted((data_dir / field_name / "snapshots").glob(f"{field_name}-*"))
    if not snapshots:
        raise FileNotFoundError(f"no snapshot for field {field_name!r} under {data_dir}")
    return snapshots[-1]


def save_snapshot(
    path: Path, nodes: pd.DataFrame, edges: pd.DataFrame, stats: GraphStats, report: CleaningReport
) -> Path:
    """Write the canonical tables plus the provenance of how they were cleaned."""
    path.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(path / "nodes.parquet", index=False)
    edges.to_parquet(path / "edges.parquet", index=False)
    (path / "stats.json").write_text(
        json.dumps({"stats": asdict(stats), "cleaning": asdict(report)}, indent=2, default=str)
    )
    return path


def load_snapshot(path: Path) -> nx.DiGraph:
    """Rebuild the graph from a saved snapshot."""
    return to_graph(
        pd.read_parquet(path / "nodes.parquet"), pd.read_parquet(path / "edges.parquet")
    )


def build_field(
    field_name: str,
    *,
    data_dir: Path = DATA_DIR,
    largest_component_only: bool = True,
) -> tuple[Path, GraphStats]:
    """Clean a field's raw crawl and write a dated snapshot."""
    config: FieldConfig = load_field(field_name)
    nodes, edges = load_raw(data_dir / config.name / "raw")
    nodes, edges, report = clean(
        nodes,
        edges,
        date_range=config.crawl.date_range,
        largest_component_only=largest_component_only,
    )
    surviving = set(nodes["id"])
    report.dropped_seeds = [seed for seed in config.seed_papers if seed not in surviving]
    report.log()

    stats = summarize(to_graph(nodes, edges))
    path = save_snapshot(snapshot_dir(data_dir, config.name), nodes, edges, stats, report)
    return path, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", default="gnn", help="field config name under fields/")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--keep-all-components",
        action="store_true",
        help="keep every component rather than only the largest",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path, stats = build_field(
        args.field,
        data_dir=args.data_dir,
        largest_component_only=not args.keep_all_components,
    )
    logger.info("snapshot written to %s", path)
    for name, value in asdict(stats).items():
        if name != "most_cited_within_graph":
            logger.info("%-28s %s", name, value)
    for node, title, degree in stats.most_cited_within_graph:
        logger.info("  %-14s %4d  %s", node, degree, title[:70])


if __name__ == "__main__":
    main()
