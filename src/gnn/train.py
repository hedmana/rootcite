"""Train the link-prediction model on the temporal splits.

Each split is evaluated on the message-passing graph it is allowed to see, and
the test split is only touched once, after the best validation checkpoint has
been restored. Selecting a checkpoint on the test edges would make the number
reported at the end a training metric wearing a disguise.
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

from gnn.dataset import LinkDataset, load_dataset
from gnn.metrics import average_precision, roc_auc
from gnn.model import LinkPredictor, ModelConfig
from graph.build import latest_snapshot
from graph.crawler import DATA_DIR

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass
class TrainConfig:
    epochs: int = 200
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    patience: int = 20
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


def _message_passing(dataset: LinkDataset) -> dict[str, Tensor]:
    """Each split's own view of the graph, made undirected once rather than per epoch."""
    count = len(dataset.node_ids)
    return {
        split: to_undirected(getattr(dataset, split).edge_index, num_nodes=count)
        for split in SPLITS
    }


def evaluate(model: LinkPredictor, split: Data, message: Tensor) -> Metrics:
    model.eval()
    with torch.no_grad():
        scores = model(split.x, message, split.edge_label_index)
    if not scores.numel():
        return Metrics(loss=float("nan"), roc_auc=float("nan"), average_precision=float("nan"))
    return Metrics(
        loss=float(functional.binary_cross_entropy_with_logits(scores, split.edge_label)),
        roc_auc=roc_auc(scores, split.edge_label),
        average_precision=average_precision(scores, split.edge_label),
    )


def _step(model: LinkPredictor, split: Data, message: Tensor, optimizer: torch.optim.Optimizer):
    model.train()
    optimizer.zero_grad()
    scores = model(split.x, message, split.edge_label_index)
    loss = functional.binary_cross_entropy_with_logits(scores, split.edge_label)
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
    message = _message_passing(dataset)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_score, epoch = 0, -math.inf, 0
    for epoch in range(1, config.epochs + 1):
        loss = _step(model, dataset.train, message["train"], optimizer)
        validation = evaluate(model, dataset.val, message["val"])

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
        train=evaluate(model, dataset.train, message["train"]),
        val=evaluate(model, dataset.val, message["val"]),
        test=evaluate(model, dataset.test, message["test"]),
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
            seed=args.seed,
        ),
    )
    logger.info("model written to %s", path)


if __name__ == "__main__":
    main()
