"""Turn a cleaned graph snapshot into a temporal link-prediction dataset.

A citation is an event with a date: it comes into existence when the citing
paper is published. Splitting those events at random would let the model see
the future while scoring the past, which is exactly the mistake an originator
ranking cannot afford to make. Every split here is a cut in time, and the
features each side of the cut sees are computed only from edges to its left.

Edges point from citing to cited work, so an edge's time is the publication
year of its source.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data

from graph.build import is_missing, latest_snapshot, load_snapshot
from graph.crawler import DATA_DIR

logger = logging.getLogger(__name__)

# `cited_by_count` is deliberately absent. OpenAlex reports it as of the crawl,
# so it counts citations made after the split point: a feature that tells the
# model which papers the future rewards is the leak the temporal split exists
# to prevent.
FEATURE_NAMES = (
    "year_scaled",
    "is_dated",
    "log_in_degree",
    "log_out_degree",
    "log_author_count",
)

YEAR, DATED, IN_DEGREE, OUT_DEGREE = (
    FEATURE_NAMES.index(name)
    for name in ("year_scaled", "is_dated", "log_in_degree", "log_out_degree")
)

UNDATED = -1
SAMPLING_ATTEMPTS = 16


@dataclass
class SplitReport:
    """How the citation record was cut in time, and what each side got."""

    nodes: int
    edges: int
    undated_nodes: int
    undated_edges: int
    train_edges: int
    val_edges: int
    test_edges: int
    train_until: int | None
    val_until: int | None
    negatives_wanted: int
    negatives_sampled: int
    seed: int

    def log(self) -> None:
        for name, value in asdict(self).items():
            logger.info("%-18s %s", name, value)
        if not self.val_edges or not self.test_edges:
            logger.warning("an evaluation split is empty; the graph may be too short-lived to cut")


@dataclass
class LinkDataset:
    """Three views of one graph, each carrying only the past it is allowed to see."""

    node_ids: list[str]
    train: Data
    val: Data
    test: Data
    report: SplitReport


def _years(graph: nx.DiGraph, node_ids: list[str]) -> Tensor:
    years = [
        UNDATED if is_missing(year := graph.nodes[node].get("publication_year")) else int(year)
        for node in node_ids
    ]
    return torch.tensor(years, dtype=torch.long)


def index_graph(graph: nx.DiGraph) -> tuple[list[str], Tensor, Tensor]:
    """Fix a node ordering, and express the citations as positions in it."""
    node_ids = list(graph.nodes)
    position = {node: order for order, node in enumerate(node_ids)}
    edges = (
        torch.tensor(
            [(position[source], position[target]) for source, target in graph.edges],
            dtype=torch.long,
        )
        .reshape(-1, 2)
        .t()
    )
    return node_ids, _years(graph, node_ids), edges


def _features(graph: nx.DiGraph, node_ids: list[str], past: Tensor, years: Tensor) -> Tensor:
    """Node features derived from the training edges alone, so nothing leaks backwards."""
    count = len(node_ids)
    dated = years != UNDATED
    known = years[dated]
    floor, span = (
        (known.min(), (known.max() - known.min()).clamp(min=1)) if known.numel() else (0, 1)
    )
    scaled = torch.where(dated, (years - floor) / span, torch.zeros(count))

    authors = torch.tensor(
        [len(graph.nodes[node].get("authors") or []) for node in node_ids], dtype=torch.float
    )
    unset = torch.zeros(count)
    features = torch.stack([scaled.float(), dated.float(), unset, unset, authors.log1p()], dim=1)
    return with_degrees(features, past)


def with_degrees(x: Tensor, edge_index: Tensor) -> Tensor:
    """`x` with its degree columns counted over `edge_index`.

    Degrees describe a graph, so they have to be counted over the one the model
    passes messages on. Counted over any other, they tell it about edges it was
    not shown.
    """
    x = x.clone()
    x[:, IN_DEGREE] = torch.bincount(edge_index[1], minlength=x.size(0)).float().log1p()
    x[:, OUT_DEGREE] = torch.bincount(edge_index[0], minlength=x.size(0)).float().log1p()
    return x


def publication_order(x: Tensor) -> Tensor:
    """Each work's place in time as its features record it, UNDATED where they record none.

    Scaled years keep the order and the ties of the years they came from, which
    is all `sample_negatives` asks of them.
    """
    return torch.where(x[:, DATED] > 0, x[:, YEAR], UNDATED)


def _cut_points(edge_years: Tensor, val_fraction: float, test_fraction: float) -> tuple[int, int]:
    """The last training year and the last validation year, by share of citations."""
    ordered = edge_years.sort().values
    count = ordered.numel()
    train_end = max(int(count * (1 - val_fraction - test_fraction)) - 1, 0)
    val_end = max(int(count * (1 - test_fraction)) - 1, 0)
    return int(ordered[train_end]), int(ordered[val_end])


def sample_negatives(
    positive: Tensor,
    years: Tensor,
    existing: Tensor,
    count: int,
    generator: torch.Generator,
    *,
    year_matched: bool = False,
) -> tuple[Tensor, Tensor]:
    """Citations each source could plausibly have made and did not.

    A negative is only informative if it was possible: the cited work has to
    already exist. Sources are drawn from the split's own positives, so the
    negatives inherit the era being scored. `year_matched` narrows the draw to
    works from the positive's cited year, so the gap between citing and cited
    says nothing about which of the two is real.

    Returns the negatives, and for each the column of `positive` it stands against.
    """
    sources, cited = positive
    dated = (years != UNDATED).nonzero(as_tuple=True)[0]
    if sources.numel() == 0 or dated.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    by_year = dated[years[dated].argsort()]
    ordered = years[by_year].contiguous()
    if year_matched:
        low = torch.searchsorted(ordered, years[cited].contiguous())
        high = torch.searchsorted(ordered, years[cited].contiguous(), right=True)
    else:
        low = torch.zeros_like(sources)
        high = torch.searchsorted(ordered, years[sources].contiguous(), right=True)

    targets = torch.zeros_like(sources)
    pending = (high > low).nonzero(as_tuple=True)[0]
    for _ in range(SAMPLING_ATTEMPTS):
        if pending.numel() == 0:
            break
        span = high[pending] - low[pending]
        draw = low[pending] + (torch.rand(pending.numel(), generator=generator) * span).long()
        candidate = by_year[draw]
        source = sources[pending]
        settled = (candidate != source) & ~torch.isin(source * count + candidate, existing)
        targets[pending[settled]] = candidate[settled]
        pending = pending[~settled]

    found = high > low
    found[pending] = False
    return torch.stack([sources[found], targets[found]]), found.nonzero(as_tuple=True)[0]


def _supervised(
    x: Tensor,
    message: Tensor,
    positive: Tensor,
    years: Tensor,
    existing: Tensor,
    count: int,
    generator: torch.Generator,
) -> Data:
    negative, _ = sample_negatives(positive, years, existing, count, generator)
    return Data(
        x=x,
        edge_index=message,
        edge_label_index=torch.cat([positive, negative], dim=1),
        edge_label=torch.cat([torch.ones(positive.size(1)), torch.zeros(negative.size(1))]),
        num_nodes=count,
    )


def build_dataset(
    graph: nx.DiGraph,
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> LinkDataset:
    """Cut a graph into train, validation and test link-prediction views."""
    node_ids, years, edges = index_graph(graph)
    count = len(node_ids)
    edge_years = years[edges[0]] if edges.numel() else torch.empty(0, dtype=torch.long)
    dated = edge_years != UNDATED

    train_until, val_until = (
        _cut_points(edge_years[dated], val_fraction, test_fraction) if dated.any() else (None, None)
    )
    is_train = dated & (edge_years <= train_until) if dated.any() else dated
    is_val = (
        dated & (edge_years > train_until) & (edge_years <= val_until) if dated.any() else dated
    )
    is_test = dated & (edge_years > val_until) if dated.any() else dated

    # Undated citations cannot be placed in time, so they never supervise, but
    # dropping them would hide real structure from message passing.
    past = edges[:, is_train | ~dated]
    x = _features(graph, node_ids, past, years)
    existing = (edges[0] * count + edges[1]).sort().values

    generator = torch.Generator().manual_seed(seed)
    splits = {
        "train": _supervised(x, past, edges[:, is_train], years, existing, count, generator),
        "val": _supervised(x, past, edges[:, is_val], years, existing, count, generator),
        "test": _supervised(
            x,
            torch.cat([past, edges[:, is_val]], dim=1),
            edges[:, is_test],
            years,
            existing,
            count,
            generator,
        ),
    }

    wanted = sum(int(split.edge_label.sum()) for split in splits.values())
    report = SplitReport(
        nodes=count,
        edges=edges.size(1),
        undated_nodes=int((years == UNDATED).sum()),
        undated_edges=int((~dated).sum()),
        train_edges=int(is_train.sum()),
        val_edges=int(is_val.sum()),
        test_edges=int(is_test.sum()),
        train_until=train_until,
        val_until=val_until,
        negatives_wanted=wanted,
        negatives_sampled=sum(int((split.edge_label == 0).sum()) for split in splits.values()),
        seed=seed,
    )
    return LinkDataset(node_ids=node_ids, report=report, **splits)


def save_dataset(path: Path, dataset: LinkDataset) -> Path:
    """Write the tensors, plus a human-readable record of where the cuts fell."""
    path.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "node_ids": dataset.node_ids,
        "feature_names": list(FEATURE_NAMES),
        "x": dataset.train.x,
    }
    for name in ("train", "val", "test"):
        split = getattr(dataset, name)
        payload[name] = {
            "edge_index": split.edge_index,
            "edge_label_index": split.edge_label_index,
            "edge_label": split.edge_label,
        }
    torch.save(payload, path / "dataset.pt")
    (path / "split.json").write_text(json.dumps(asdict(dataset.report), indent=2))
    return path


def load_dataset(path: Path) -> LinkDataset:
    """Rebuild a saved dataset without unpickling anything but tensors and text."""
    payload = torch.load(path / "dataset.pt", weights_only=True)
    node_ids = payload["node_ids"]
    splits = {
        name: Data(x=payload["x"], num_nodes=len(node_ids), **payload[name])
        for name in ("train", "val", "test")
    }
    report = SplitReport(**json.loads((path / "split.json").read_text()))
    return LinkDataset(node_ids=node_ids, report=report, **splits)


def prepare_field(
    field_name: str,
    *,
    data_dir: Path = DATA_DIR,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[Path, SplitReport]:
    """Build a dataset from a field's most recent snapshot, beside that snapshot."""
    snapshot = latest_snapshot(data_dir, field_name)
    dataset = build_dataset(
        load_snapshot(snapshot),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    dataset.report.log()
    return save_dataset(snapshot, dataset), dataset.report


def observed_view(graph: nx.DiGraph) -> tuple[list[str], Data]:
    """Every node and every citation on record, featurised the way training was.

    The temporal split exists to measure the model honestly. Scoring is not
    measurement: nothing is held back, because nothing is being tested.
    """
    node_ids, years, edges = index_graph(graph)
    return node_ids, Data(
        x=_features(graph, node_ids, edges, years), edge_index=edges, num_nodes=len(node_ids)
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", default="gnn", help="field config name under fields/")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path, _ = prepare_field(
        args.field,
        data_dir=args.data_dir,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    logger.info("dataset written to %s", path)


if __name__ == "__main__":
    main()
