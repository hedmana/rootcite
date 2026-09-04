import random

import networkx as nx
import torch
from torch_geometric.utils import to_undirected

from gnn.dataset import build_dataset
from gnn.model import ModelConfig
from gnn.train import TrainConfig, load_model, save_model, train


def preferential_attachment(size=300, per_year=10, links=4, seed=0):
    """Papers cite earlier work in proportion to how often it has been cited.

    A random citation graph has nothing to learn; this one has one honest
    signal, so a model that reports skill on it is reporting something real.
    """
    rng = random.Random(seed)
    graph = nx.DiGraph()
    popularity: dict[str, float] = {}
    for position in range(size):
        node = f"W{position}"
        graph.add_node(node, publication_year=2000 + position // per_year, authors=["Ada Lovelace"])
        popularity[node] = 1.0
        earlier = [f"W{other}" for other in range(position - position % per_year)]
        if not earlier:
            continue
        weights = [popularity[candidate] for candidate in earlier]
        for target in set(rng.choices(earlier, weights=weights, k=links)):
            graph.add_edge(node, target)
            popularity[target] += 3.0
    return graph


def dataset():
    return build_dataset(preferential_attachment(), seed=1)


def test_the_model_learns_a_signal_that_survives_the_split():
    _, report = train(dataset(), config=TrainConfig(epochs=80, patience=20))

    assert report.test.roc_auc > 0.7
    assert report.test.average_precision > 0.7


def test_training_stops_once_validation_stops_improving():
    config = TrainConfig(epochs=200, patience=2)

    _, report = train(dataset(), config=config)

    assert report.epochs_run < config.epochs
    assert report.best_epoch <= report.epochs_run


def test_the_restored_checkpoint_is_the_best_one_not_the_last():
    model, report = train(dataset(), config=TrainConfig(epochs=60, patience=10))

    replayed = train(dataset(), config=TrainConfig(epochs=report.best_epoch, patience=10))[0]

    assert torch.allclose(model.decoder.bias, replayed.decoder.bias)


def test_the_seed_fixes_the_run():
    """Two runs of one seed agree to within the order a parallel scatter sums in."""
    config = TrainConfig(epochs=20, patience=20, seed=4)

    first, _ = train(dataset(), config=config)
    again, _ = train(dataset(), config=config)
    other, _ = train(dataset(), config=TrainConfig(epochs=20, patience=20, seed=5))

    assert torch.allclose(first.decoder.citing.weight, again.decoder.citing.weight)
    assert not torch.allclose(first.decoder.citing.weight, other.decoder.citing.weight)


def test_a_checkpoint_round_trip_scores_identically(tmp_path):
    split = dataset()
    model, report = train(split, config=TrainConfig(epochs=20, patience=20))

    reloaded = load_model(save_model(tmp_path, model, report))

    message = to_undirected(split.test.edge_index, num_nodes=len(split.node_ids))
    model.eval()
    with torch.no_grad():
        assert torch.allclose(
            model(split.test.x, message, split.test.edge_label_index),
            reloaded(split.test.x, message, split.test.edge_label_index),
        )


def test_the_report_records_how_the_run_was_configured(tmp_path):
    config = TrainConfig(epochs=15, patience=5, seed=2)
    model_config = ModelConfig(hidden_channels=16, out_channels=8, layers=2, dropout=0.1)

    _, report = train(dataset(), model_config=model_config, config=config)

    assert report.model["out_channels"] == 8
    assert report.training["seed"] == 2
