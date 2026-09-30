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
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from graph.config import DateRange, FieldConfig, load_field
from graph.crawler import DATA_DIR

logger = logging.getLogger(__name__)

# A preprint can be cited a year before the venue date OpenAlex gives it.
PREPRINT_SLACK = 1

# A preprint and its venue version are dated at most this many years apart.
VERSION_GAP = 2

# Citations `graph.semanticscholar` recovered, kept apart from the crawl's own.
RECOVERED_EDGES = "recovered_edges.parquet"

# Library catalogues file a person as `Last, First 1968-`, which a list of names
# reads as two people.
LIFE_DATES = re.compile(r"\s+\d{4}-(\d{4})?$")
FAMILY_NAME_FIRST = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")

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
    excluded_nodes: int = 0
    retitled_nodes: int = 0
    merged_nodes: int = 0
    renamed_authors: int = 0
    duplicate_nodes: int = 0
    duplicate_edges: int = 0
    out_of_range_nodes: int = 0
    dangling_edges: int = 0
    redated_nodes: int = 0
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


def display_name(name: str) -> str:
    """`Schölkopf, Bernhard 1968-` as `Bernhard Schölkopf`; a family-name-first script keeps it."""
    name = LIFE_DATES.sub("", name)
    if name.count(",") != 1:
        return name
    family, given = (part.strip() for part in name.split(","))
    ordered = (family, given) if FAMILY_NAME_FIRST.search(name) else (given, family)
    return " ".join(part for part in ordered if part)


def _rename_authors(authors: pd.Series) -> tuple[pd.Series, int]:
    renamed = 0

    def rename(names: Iterable[str] | None) -> list[str] | None:
        nonlocal renamed
        if names is None:
            return None
        shown = [display_name(name) for name in names]
        renamed += sum(old != new for old, new in zip(names, shown, strict=True))
        return shown

    return authors.map(rename), renamed


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the crawler's raw node and edge tables, with any recovered citations."""
    nodes = pd.read_parquet(raw_dir / "nodes.parquet")
    edges = pd.read_parquet(raw_dir / "edges.parquet")
    if (recovered := raw_dir / RECOVERED_EDGES).exists():
        edges = pd.concat([edges, pd.read_parquet(recovered)], ignore_index=True)
    return nodes, edges


def clean(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    date_range: DateRange | None = None,
    exclude: Iterable[str] = (),
    retitle: Mapping[str, str] | None = None,
    merge: Mapping[str, str] | None = None,
    largest_component_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Reduce raw crawl tables to a consistent node set and edge set."""
    report = CleaningReport(raw_nodes=len(nodes), raw_edges=len(edges))

    nodes = nodes.drop_duplicates(subset="id")
    report.duplicate_nodes = report.raw_nodes - len(nodes)

    excluded = nodes["id"].isin(set(exclude))
    report.excluded_nodes = int(excluded.sum())
    nodes = nodes[~excluded].copy()

    # The abstract came from the same wrong record as the title, so it goes too.
    wrong = nodes["id"].isin(set(retitle or {}))
    report.retitled_nodes = int(wrong.sum())
    nodes.loc[wrong, "title"] = nodes.loc[wrong, "id"].map(retitle or {})
    if "abstract" in nodes:
        nodes.loc[wrong, "abstract"] = None

    if "authors" in nodes:
        nodes["authors"], report.renamed_authors = _rename_authors(nodes["authors"])

    held = len(nodes)
    nodes, edges = _fold(nodes, edges, merge or {})
    nodes, edges = _fold(nodes, edges, _versions(nodes, edges))
    report.merged_nodes = held - len(nodes)

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
    nodes, report.redated_nodes = _redate(nodes, edges)

    if largest_component_only:
        keep = _largest_component(nodes["id"], edges)
        report.disconnected_nodes = len(nodes) - len(keep)
        nodes = nodes[nodes["id"].isin(keep)]
        edges = edges[edges["source"].isin(keep) & edges["target"].isin(keep)]

    report.nodes = len(nodes)
    report.edges = len(edges)
    return nodes, edges, report


def _fold(
    nodes: pd.DataFrame, edges: pd.DataFrame, duplicates: Mapping[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold each duplicate record into the one it duplicates, citations and all."""
    held = set(nodes["id"])
    duplicates = {dup: kept for dup, kept in duplicates.items() if dup in held and kept in held}
    if not duplicates:
        return nodes, edges

    edges = edges.assign(
        source=edges["source"].map(duplicates).fillna(edges["source"]),
        target=edges["target"].map(duplicates).fillna(edges["target"]),
    )
    folded = nodes["id"].isin(duplicates.keys())
    if "abstract" in nodes:
        spare = nodes[folded & ~nodes["abstract"].map(is_missing)]
        spare = spare.set_index(spare["id"].map(duplicates))["abstract"]
        spare = spare[~spare.index.duplicated()]
        lacking = ~folded & nodes["abstract"].map(is_missing) & nodes["id"].isin(spare.index)
        nodes = nodes.copy()
        nodes.loc[lacking, "abstract"] = nodes.loc[lacking, "id"].map(spare)
    return nodes[~folded], edges


def _versions(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, str]:
    """Preprint and venue records of one work, each mapped to the one the crawl cites most.

    They share a title, a first author's family name and roughly a year. A title
    alone is not enough: `Deep learning` is both LeCun et al. 2015 and Goodfellow
    et al. 2016. A given name is too much: DiffPool's first author is both Rex and
    Zhitao Ying.
    """
    if "authors" not in nodes or nodes.empty:
        return {}
    # A version OpenAlex dates by a late reprint is dated by its citers instead.
    records, _ = _redate(nodes, edges)
    records = records.assign(
        key=records["title"].map(_title_key),
        first=records["authors"].map(_first_author),
        cited=records["id"].map(edges["target"].value_counts()).fillna(0),
    )
    records = records[(records["key"] != "") & (records["first"] != "")]
    records = records[records.duplicated(["key", "first"], keep=False)]
    duplicates = {}
    for _, group in records.sort_values(["cited", "id"], ascending=[False, True]).groupby(
        ["key", "first"], sort=False
    ):
        kept, *others = group.itertuples(index=False)
        for other in others:
            if abs(other.publication_year - kept.publication_year) <= VERSION_GAP:
                duplicates[other.id] = kept.id
    return duplicates


def _title_key(title: object) -> str:
    return "" if is_missing(title) else re.sub(r"[^a-z0-9]", "", str(title).lower())


def _first_author(names: object) -> str:
    """The first author's family name; a name with no Latin letters as written."""
    if not isinstance(names, list | tuple | np.ndarray) or len(names) == 0:
        return ""
    name = str(names[0])
    words = re.findall(
        r"[a-z]+", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    )
    return words[-1] if words else "".join(name.split())


def _redate(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Pull back any work dated well after the works that cite it.

    OpenAlex sometimes dates a work by a late reprint or a mis-merged record. A
    work cited in 2017 but dated 2025 turns every citation of it into one of the
    future, which a temporal split reads as a free label. Such a work takes the
    year of its earliest citer.
    """
    if edges.empty:
        return nodes, 0
    year = nodes.set_index("id")["publication_year"]
    earliest = nodes["id"].map(edges["source"].map(year).groupby(edges["target"]).min())
    late = nodes["publication_year"] > earliest + PREPRINT_SLACK
    nodes = nodes.copy()
    nodes.loc[late, "publication_year"] = earliest[late].astype(nodes["publication_year"].dtype)
    return nodes, int(late.sum())


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
        exclude=config.exclude_works,
        retitle=config.corrected_titles,
        merge=config.duplicate_works,
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
