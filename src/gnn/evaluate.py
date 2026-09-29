"""Measure the link model against the heuristics it has to beat.

`gnn.train` reports how well the model separates real citations from sampled
ones, and on its own that number says little. Papers mostly cite recent work,
so when non-citations are drawn from everything earlier, the year gap alone
separates them well. Every score here is therefore set beside heuristics fitted
to the same task, under two samplers: uniform over earlier work, and matched to
the cited work's year, where the gap carries no signal and only structure can.

Beside AUC, each true citation is ranked among non-citations drawn for it alone,
the way a reading list would have to rank it. And no run stands alone: seeds
vary the negatives and the initialisation, and rolling cuts move the test window
back through time. A gap smaller than the spread across runs is not a gap.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import networkx as nx
import torch
from torch import Tensor
from torch.nn import functional
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from gnn.dataset import build_dataset, index_graph, sample_negatives
from gnn.metrics import positive_ranks, roc_auc
from gnn.model import LinkPredictor
from gnn.train import TrainConfig, train
from graph.build import is_missing, latest_snapshot, load_snapshot
from graph.crawler import DATA_DIR

logger = logging.getLogger(__name__)

SAMPLERS = {"uniform": False, "year_matched": True}
METRICS = ("roc_auc", "mrr", "hits_at_10")

Scorer = Callable[[Tensor], Tensor]
Row = dict[str, float | int | str | None]


def rolling_cuts(graph: nx.DiGraph, cuts: int, window: float = 0.1) -> list[nx.DiGraph]:
    """The graph as it stood before each of its last `cuts` windows of citations.

    Each view ends a further `window` share of dated citations earlier, so its
    test split falls that much further back. Cuts that land on the same year are
    the same view, and are kept once. Undated work stays in every view: it has
    no place in time to be cut from.
    """

    def year(node: str) -> float | None:
        return graph.nodes[node].get("publication_year")

    edge_years = sorted(year(source) for source, _ in graph.edges if not is_missing(year(source)))
    horizons = dict.fromkeys(
        edge_years[int(len(edge_years) * (1 - k * window)) - 1] for k in range(cuts)
    )
    return [
        graph.subgraph(node for node in graph if is_missing(year(node)) or year(node) <= horizon)
        for horizon in horizons
    ]


def _design(pairs: Tensor, years: Tensor, in_degree: Tensor) -> Tensor:
    """What a citation's odds need no model for: the gap it spans, how cited its target is."""
    source, cited = pairs
    gap = (years[source] - years[cited]).float()
    return torch.stack([gap, in_degree[cited].float().log1p(), torch.ones_like(gap)], dim=1)


def _fit_logistic(design: Tensor, labels: Tensor) -> Tensor:
    weights = torch.zeros(design.size(1), requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], max_iter=100, line_search_fn="strong_wolfe")

    def loss() -> Tensor:
        optimizer.zero_grad()
        value = functional.binary_cross_entropy_with_logits(design @ weights, labels)
        value.backward()
        return value

    optimizer.step(loss)
    return weights.detach()


def _scorers(
    model: LinkPredictor, z: Tensor, years: Tensor, in_degree: Tensor, weights: Tensor
) -> dict[str, Scorer]:
    return {
        "model": lambda pairs: model.decode(z, pairs),
        "recency": lambda pairs: (years[pairs[1]] - years[pairs[0]]).float(),
        "in_degree": lambda pairs: in_degree[pairs[1]].float(),
        "logistic": lambda pairs: _design(pairs, years, in_degree) @ weights,
    }


def _measure(score: Scorer, positive: Tensor, negative: Tensor, owner: Tensor) -> dict[str, float]:
    with torch.no_grad():
        hit, miss = score(positive), score(negative)
    # A positive the sampler found no rival for has no rank to report.
    ranks = positive_ranks(hit, miss, owner)[torch.bincount(owner, minlength=hit.numel()) > 0]
    return {
        "roc_auc": roc_auc(
            torch.cat([hit, miss]), torch.cat([torch.ones_like(hit), torch.zeros_like(miss)])
        ),
        "mrr": float(ranks.reciprocal().mean()),
        "hits_at_10": float((ranks <= 10).float().mean()),
    }


def evaluate_cut(
    graph: nx.DiGraph, *, seed: int = 0, candidates: int = 100, config: TrainConfig | None = None
) -> list[Row]:
    """Train once on this view of the graph, then score its test window every way there is.

    The logistic baseline is fitted on the validation split, the same data the
    model's checkpoint is chosen on, so neither side has seen the test window.
    """
    dataset = build_dataset(graph, seed=seed)
    model, _ = train(dataset, config=replace(config or TrainConfig(), seed=seed))

    _, years, edges = index_graph(graph)
    count = len(dataset.node_ids)
    existing = (edges[0] * count + edges[1]).sort().values
    generator = torch.Generator().manual_seed(seed)

    def contest(split: Data, rivals: int, year_matched: bool) -> tuple[Tensor, Tensor, Tensor]:
        positive = split.edge_label_index[:, split.edge_label == 1]
        negative, owner = sample_negatives(
            positive.repeat_interleave(rivals, dim=1),
            years,
            existing,
            count,
            generator,
            year_matched=year_matched,
        )
        return positive, negative, owner // rivals

    in_degree = {
        name: torch.bincount(getattr(dataset, name).edge_index[1], minlength=count)
        for name in ("val", "test")
    }
    model.eval()
    with torch.no_grad():
        z = model.encode(dataset.test.x, to_undirected(dataset.test.edge_index, num_nodes=count))

    rows: list[Row] = []
    for sampler, year_matched in SAMPLERS.items():
        positive, negative, _ = contest(dataset.val, 1, year_matched)
        weights = _fit_logistic(
            _design(torch.cat([positive, negative], dim=1), years, in_degree["val"]),
            torch.cat([torch.ones(positive.size(1)), torch.zeros(negative.size(1))]),
        )
        positive, negative, owner = contest(dataset.test, candidates, year_matched)
        for scorer, score in _scorers(model, z, years, in_degree["test"], weights).items():
            rows.append(
                {
                    "tested_after": dataset.report.val_until,
                    "seed": seed,
                    "sampler": sampler,
                    "scorer": scorer,
                    **_measure(score, positive, negative, owner),
                }
            )
    return rows


def summarise(rows: list[Row]) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Mean and spread of every metric, per sampler and scorer, across all runs."""
    grouped: dict[str, dict[str, list[Row]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sampler"]), {}).setdefault(str(row["scorer"]), []).append(row)

    def spread(values: list) -> dict[str, float]:
        tensor = torch.tensor(values, dtype=torch.float64)
        return {"mean": float(tensor.mean()), "std": float(tensor.std(correction=0))}

    return {
        sampler: {
            scorer: {metric: spread([run[metric] for run in runs]) for metric in METRICS}
            for scorer, runs in scorers.items()
        }
        for sampler, scorers in grouped.items()
    }


def log_summary(summary: dict[str, dict[str, dict[str, dict[str, float]]]]) -> None:
    logger.info("%-13s %-10s%s", "sampler", "scorer", "".join(f"{m:>18}" for m in METRICS))
    for sampler, scorers in summary.items():
        for scorer, metrics in scorers.items():
            cells = "".join(
                f"{metrics[m]['mean']:>11.3f} ± {metrics[m]['std']:.3f}" for m in METRICS
            )
            logger.info("%-13s %-10s%s", sampler, scorer, cells)


def evaluate_field(
    field_name: str,
    *,
    data_dir: Path = DATA_DIR,
    seeds: int = 5,
    cuts: int = 3,
    candidates: int = 100,
) -> tuple[Path, dict[str, dict[str, dict[str, dict[str, float]]]]]:
    """Every seed on every cut of a field's latest snapshot, written beside it."""
    snapshot = latest_snapshot(data_dir, field_name)
    rows: list[Row] = []
    for view in rolling_cuts(load_snapshot(snapshot), cuts):
        for seed in range(seeds):
            rows.extend(evaluate_cut(view, seed=seed, candidates=candidates))
            logger.info("tested after %s, seed %d", rows[-1]["tested_after"], seed)

    summary = summarise(rows)
    path = snapshot / "evaluation.json"
    path.write_text(
        json.dumps({"candidates": candidates, "summary": summary, "runs": rows}, indent=2)
    )
    return path, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", default="gnn", help="field config name under fields/")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--cuts", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=100, help="negatives per positive")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path, summary = evaluate_field(
        args.field,
        data_dir=args.data_dir,
        seeds=args.seeds,
        cuts=args.cuts,
        candidates=args.candidates,
    )
    log_summary(summary)
    logger.info("evaluation written to %s", path)


if __name__ == "__main__":
    main()
