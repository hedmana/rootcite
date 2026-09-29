"""Train the link-prediction model on the temporal splits.

Each split is evaluated on the message-passing graph it is allowed to see, and
the test split is only touched once, after the best validation checkpoint has
been restored. Selecting a checkpoint on the test edges would make the number
reported at the end a training metric wearing a disguise.

Training has to pose the question evaluation does. A validation or test
citation comes from a paper the message graph has never met: nothing it cites
and nothing citing it is on record yet. Supervised on citations the graph
already holds, a model learns to recognise edges it can see, and that does not
carry over. So every epoch cuts a share of the citing papers out of the graph
and scores only their citations, against negatives drawn afresh.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from gnn.dataset import (
    LinkDataset,
    load_dataset,
    publication_order,
    sample_negatives,
    with_degrees,
)
from gnn.metrics import average_precision, roc_auc
from gnn.model import LinkPredictor, ModelConfig
from graph.build import latest_snapshot
from graph.crawler import DATA_DIR

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass
class TrainConfig:
    epochs: int = 600
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    # One step from a random start can rank by popularity alone, which a dense
    # graph rewards; learning more dips below that first, so patience has to
    # outlast the dip.
    patience: int = 100
    held_out: float = 0.2
    seed: int = 0


@dataclass
class Metrics:
    loss: float
    roc_auc: float
    average_precision: float


@dataclass
class TrainingReport:
    epochs_run: int
    best_epoch: int
    train: Metrics
    val: Metrics
    test: Metrics
    model: dict[str, float | int]
    training: dict[str, float | int]

    def log(self) -> None:
        for split in SPLITS:
            metrics = getattr(self, split)
            logger.info(
                "%-6s loss %.4f  auc %.4f  ap %.4f",
                split,
                metrics.loss,
                metrics.roc_auc,
                metrics.average_precision,
            )


View = tuple[Tensor, Tensor]


def _view(x: Tensor, edge_index: Tensor) -> View:
    """Features and undirected message graph, both taken from the same citations."""
    return with_degrees(x, edge_index), to_undirected(edge_index, num_nodes=x.size(0))


def _views(dataset: LinkDataset) -> dict[str, View]:
    """Each split's own view of the graph, built once rather than per epoch."""
    return {
        split: _view(getattr(dataset, split).x, getattr(dataset, split).edge_index)
        for split in SPLITS
    }


def _record(dataset: LinkDataset) -> Tensor:
    """Every citation on record, as sorted keys, so no negative is one that happened.

    The test split sees everything but its own positives, so the two together
    are the whole record.
    """
    test = dataset.test
    edges = torch.cat([test.edge_index, test.edge_label_index[:, test.edge_label == 1]], dim=1)
    return (edges[0] * len(dataset.node_ids) + edges[1]).sort().values


def _cold_start(
    positive: Tensor, past: Tensor, share: float, count: int, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """One epoch's supervision, and the graph the model may see while scoring it."""
    sources = positive[0].unique()
    held = torch.zeros(count, dtype=torch.bool)
    held[sources[torch.rand(sources.numel(), generator=generator) < share]] = True
    return positive[:, held[positive[0]]], past[:, ~(held[past[0]] | held[past[1]])]


def evaluate(model: LinkPredictor, split: Data, view: View) -> Metrics:
    model.eval()
    with torch.no_grad():
        scores = model(*view, split.edge_label_index)
    if not scores.numel():
        return Metrics(loss=float("nan"), roc_auc=float("nan"), average_precision=float("nan"))
    return Metrics(
        loss=float(functional.binary_cross_entropy_with_logits(scores, split.edge_label)),
        roc_auc=roc_auc(scores, split.edge_label),
        average_precision=average_precision(scores, split.edge_label),
    )


def _step(
    model: LinkPredictor,
    view: View,
    positive: Tensor,
    negative: Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad()
    scores = model(*view, torch.cat([positive, negative], dim=1))
    labels = torch.cat([torch.ones(positive.size(1)), torch.zeros(negative.size(1))])
    loss = functional.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def train(
    dataset: LinkDataset,
    *,
    model_config: ModelConfig | None = None,
    config: TrainConfig | None = None,
) -> tuple[LinkPredictor, TrainingReport]:
    """Fit until validation precision stops improving, then score the future once."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)

    model = LinkPredictor(dataset.train.x.size(1), model_config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    views = _views(dataset)
    count = len(dataset.node_ids)
    train_split = dataset.train
    positive = train_split.edge_label_index[:, train_split.edge_label == 1]
    order, record = publication_order(train_split.x), _record(dataset)
    generator = torch.Generator().manual_seed(config.seed)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_score, epoch = 0, -math.inf, 0
    for epoch in range(1, config.epochs + 1):
        supervised, message = _cold_start(
            positive, train_split.edge_index, config.held_out, count, generator
        )
        negative, _ = sample_negatives(supervised, order, record, count, generator)
        loss = _step(model, _view(train_split.x, message), supervised, negative, optimizer)
        validation = evaluate(model, dataset.val, views["val"])

        # An empty validation split has nothing to rank, so fit is all there is to go on.
        score = validation.average_precision
        if not math.isfinite(score):
            score = -loss

        if score > best_score:
            best_state, best_epoch, best_score = copy.deepcopy(model.state_dict()), epoch, score
        elif epoch - best_epoch >= config.patience:
            break

    model.load_state_dict(best_state)
    report = TrainingReport(
        epochs_run=epoch,
        best_epoch=best_epoch,
        train=evaluate(model, dataset.train, views["train"]),
        val=evaluate(model, dataset.val, views["val"]),
        test=evaluate(model, dataset.test, views["test"]),
        model=asdict(model.config),
        training=asdict(config),
    )
    return model, report


def save_model(path: Path, model: LinkPredictor, report: TrainingReport) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_channels": model.encoder.normalize.num_features,
            "config": asdict(model.config),
        },
        path / "model.pt",
    )
    (path / "metrics.json").write_text(json.dumps(asdict(report), indent=2))
    return path


def load_model(path: Path) -> LinkPredictor:
    checkpoint = torch.load(path / "model.pt", weights_only=True)
    model = LinkPredictor(checkpoint["in_channels"], ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def train_field(
    field_name: str,
    *,
    data_dir: Path = DATA_DIR,
    model_config: ModelConfig | None = None,
    config: TrainConfig | None = None,
) -> tuple[Path, TrainingReport]:
    """Train on a field's most recent dataset and check the model in beside it."""
    snapshot = latest_snapshot(data_dir, field_name)
    model, report = train(load_dataset(snapshot), model_config=model_config, config=config)
    report.log()
    return save_model(snapshot, model, report), report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", default="gnn", help="field config name under fields/")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--hidden-channels", type=int, default=ModelConfig.hidden_channels)
    parser.add_argument("--out-channels", type=int, default=ModelConfig.out_channels)
    parser.add_argument("--layers", type=int, default=ModelConfig.layers)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument(
        "--held-out",
        type=float,
        default=TrainConfig.held_out,
        help="share of citing papers cut from the graph each epoch",
    )
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path, _ = train_field(
        args.field,
        data_dir=args.data_dir,
        model_config=ModelConfig(
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            layers=args.layers,
            dropout=args.dropout,
        ),
        config=TrainConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            held_out=args.held_out,
            seed=args.seed,
        ),
    )
    logger.info("model written to %s", path)


if __name__ == "__main__":
    main()
