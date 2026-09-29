"""Judge originator rankings against lineages a person has checked.

Link prediction says whether the model understands citations. It does not say
whether the ranking a reader sees names the works that made a paper possible.
That takes an answer key, `fields/gold/<field>.yaml`: papers, and for each the
earlier works it grew from. Every scorer ranks each paper's ancestry and is
judged on how many of those works reach its top ten, and how high.

A gold originator the crawl never reached, or reached by no path from its paper,
cannot be ranked by anything. It counts against coverage, reported on its own,
rather than against a scorer.

Two further checks ask whether a ranking can be trusted at all. The learned one
should not change with the seed its model was trained from, and no ranking
should change much when a tenth of the citations go missing, since OpenAlex
already loses about that many.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from itertools import combinations
from pathlib import Path
from typing import Literal

import networkx as nx
import yaml
from pydantic import BaseModel
from torch import Tensor

from gnn.content import Content
from gnn.dataset import build_dataset
from gnn.model import LinkPredictor
from gnn.originators import embed, flow, learned_flow
from gnn.train import TrainConfig, load_model, train
from graph.build import is_missing, latest_snapshot, load_snapshot
from graph.config import FIELDS_DIR
from graph.crawler import DATA_DIR
from graph.score import SCORERS, acyclic, ancestry, rank

logger = logging.getLogger(__name__)

GOLD_DIR = FIELDS_DIR / "gold"


class Lineage(BaseModel):
    target: str
    # `dev` papers may steer changes to the ranking; `test` papers only report it.
    split: Literal["dev", "test"]
    # One entry per work, as every OpenAlex id it goes by: any of them counts.
    originators: list[list[str]]


class Gold(BaseModel):
    lineages: list[Lineage]


def load_gold(path: Path) -> list[Lineage]:
    if not path.is_file():
        raise FileNotFoundError(f"no gold lineages at {path}")
    return Gold.model_validate(yaml.safe_load(path.read_text())).lineages


def rankings(
    graph: nx.DiGraph,
    target: str,
    model: LinkPredictor,
    embedding: tuple[dict[str, int], Tensor],
    content: Content,
    top: int,
) -> dict[str, list[str]]:
    """Every scorer's top works for one paper: as served, without content, unweighted, baselines."""
    scores = {
        "learned": learned_flow(graph, target, model, embedding=embedding, content=content),
        "model_only": learned_flow(graph, target, model, embedding=embedding),
        "uniform_flow": flow(acyclic(ancestry(graph, target)), target),
        **{name: scorer(graph, target) for name, scorer in SCORERS.items()},
    }
    return {name: [node for node, _ in rank(ranked, top)] for name, ranked in scores.items()}


def judge(ranking: list[str], findable: list[list[str]], top: int) -> tuple[float, float]:
    """Recall and nDCG of a top-`top` ranking, over the gold originators it could have found."""
    if not findable:
        return math.nan, math.nan
    work = {work_id: order for order, ids in enumerate(findable) for work_id in ids}
    found: set[int] = set()
    gain = 0.0
    for position, node in enumerate(ranking, start=1):
        if (order := work.get(node)) is not None and order not in found:
            found.add(order)
            gain += 1 / math.log2(position + 1)
    ideal = sum(1 / math.log2(position + 1) for position in range(1, min(len(findable), top) + 1))
    return len(found) / len(findable), gain / ideal


def _overlap(left: list[str], right: list[str]) -> float:
    return len(set(left) & set(right)) / len(left) if left else math.nan


def _thinned(graph: nx.DiGraph, share: float, seed: int) -> nx.DiGraph:
    """The graph with a random `share` of its citations gone."""
    chance = random.Random(seed)
    thinned = nx.DiGraph()
    thinned.add_nodes_from(graph.nodes(data=True))
    thinned.add_edges_from(edge for edge in graph.edges if chance.random() >= share)
    return thinned


def _mean(values) -> float:
    kept = [value for value in values if not math.isnan(value)]
    return sum(kept) / len(kept) if kept else math.nan


def assess(
    graph: nx.DiGraph,
    gold: list[Lineage],
    model: LinkPredictor,
    *,
    top: int = 10,
    seeds: int = 3,
    dropout: float = 0.1,
) -> dict:
    """Score, cover and stress every scorer on every gold paper the graph holds."""
    lineages = [lineage for lineage in gold if lineage.target in graph]
    for lineage in gold:
        if lineage.target not in graph:
            logger.warning("gold target %s is not in the graph", lineage.target)

    embedding = embed(graph, model)
    content = Content(graph)
    thinned = _thinned(graph, dropout, seed=0)
    thinned_embedding = embed(thinned, model)
    others = [
        train(build_dataset(graph, seed=seed), config=TrainConfig(seed=seed))[0]
        for seed in range(1, seeds)
    ]
    other_embeddings = [embed(graph, other) for other in others]

    targets = []
    for lineage in lineages:
        reachable = set(ancestry(graph, lineage.target))
        findable = [ids for ids in lineage.originators if reachable.intersection(ids)]
        ranked = rankings(graph, lineage.target, model, embedding, content, top)
        stressed = rankings(thinned, lineage.target, model, thinned_embedding, content, top)
        learned = [ranked["learned"]] + [
            [
                node
                for node, _ in rank(
                    learned_flow(graph, lineage.target, other, embedding=e, content=content), top
                )
            ]
            for other, e in zip(others, other_embeddings, strict=True)
        ]
        title = graph.nodes[lineage.target].get("title")
        targets.append(
            {
                "target": lineage.target,
                "split": lineage.split,
                "title": None if is_missing(title) else str(title),
                "originators": len(lineage.originators),
                "reachable": len(findable),
                "seed_overlap": _mean(_overlap(a, b) for a, b in combinations(learned, 2)),
                "scorers": {
                    name: dict(
                        zip(("recall", "ndcg"), judge(ranking, findable, top), strict=True),
                        dropout_overlap=_overlap(ranking, stressed[name]),
                        found=sum(bool(set(ranking) & set(ids)) for ids in lineage.originators),
                        top=ranking,
                    )
                    for name, ranking in ranked.items()
                },
            }
        )

    return {
        "top": top,
        "targets_in_graph": f"{len(lineages)}/{len(gold)}",
        "splits": {
            split: _summarise([t for t in targets if t["split"] == split])
            for split in ("dev", "test")
        },
        "targets": targets,
    }


def _summarise(targets: list[dict]) -> dict:
    """Means over papers, except gold found, which is counted over every originator."""
    names = targets[0]["scorers"] if targets else {}
    reachable, originators = (sum(t[key] for t in targets) for key in ("reachable", "originators"))
    return {
        "coverage": f"{reachable}/{originators}",
        "seed_overlap": _mean(t["seed_overlap"] for t in targets),
        "scorers": {
            name: {
                **{
                    metric: _mean(t["scorers"][name][metric] for t in targets)
                    for metric in ("recall", "ndcg", "dropout_overlap")
                },
                "found": f"{sum(t['scorers'][name]['found'] for t in targets)}/{originators}",
            }
            for name in names
        },
    }


def log_assessment(result: dict) -> None:
    logger.info("gold targets in graph %s", result["targets_in_graph"])
    for split, summary in result["splits"].items():
        logger.info(
            "\n%s: originators reachable %s, learned top-%d overlap across seeds %.3f",
            split,
            summary["coverage"],
            result["top"],
            summary["seed_overlap"],
        )
        logger.info("%-22s %8s %8s %8s %10s", "scorer", "found", "recall", "ndcg", "dropout")
        for name, metrics in summary["scorers"].items():
            logger.info(
                "%-22s %8s %8.3f %8.3f %10.3f",
                name,
                metrics["found"],
                metrics["recall"],
                metrics["ndcg"],
                metrics["dropout_overlap"],
            )


def assess_field(
    field_name: str,
    *,
    data_dir: Path = DATA_DIR,
    gold_path: Path | None = None,
    top: int = 10,
    seeds: int = 3,
    dropout: float = 0.1,
) -> tuple[Path, dict]:
    """Judge a field's served snapshot and model, and write the verdict beside them."""
    snapshot = latest_snapshot(data_dir, field_name)
    result = assess(
        load_snapshot(snapshot),
        load_gold(gold_path or GOLD_DIR / f"{field_name}.yaml"),
        load_model(snapshot),
        top=top,
        seeds=seeds,
        dropout=dropout,
    )
    path = snapshot / "lineages.json"
    path.write_text(json.dumps(result, indent=2))
    return path, result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", default="gnn", help="field config name under fields/")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--gold", type=Path, default=None, help="defaults to fields/gold/<field>")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3, help="models trained to test stability")
    parser.add_argument("--dropout", type=float, default=0.1, help="share of citations removed")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path, result = assess_field(
        args.field,
        data_dir=args.data_dir,
        gold_path=args.gold,
        top=args.top,
        seeds=args.seeds,
        dropout=args.dropout,
    )
    log_assessment(result)
    logger.info("verdict written to %s", path)


if __name__ == "__main__":
    main()
