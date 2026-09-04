import torch

from gnn.model import LinkPredictor, ModelConfig


def graph_batch(nodes=8, features=5):
    x = torch.randn(nodes, features)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
    return x, edge_index


def test_scoring_a_citation_is_not_symmetric():
    """A cites B is a different claim from B cites A, and must score differently."""
    torch.manual_seed(0)
    x, edge_index = graph_batch()
    model = LinkPredictor(x.size(1)).eval()

    z = model.encode(x, edge_index)
    forward = model.decode(z, torch.tensor([[0], [5]]))
    backward = model.decode(z, torch.tensor([[5], [0]]))

    assert not torch.allclose(forward, backward)


def test_one_score_comes_back_per_candidate_edge():
    x, edge_index = graph_batch()
    candidates = torch.tensor([[0, 1, 2], [4, 5, 6]])

    scores = LinkPredictor(x.size(1)).eval()(x, edge_index, candidates)

    assert scores.shape == (candidates.size(1),)


def test_the_configured_depth_is_the_number_of_layers():
    model = LinkPredictor(5, ModelConfig(layers=3, hidden_channels=7, out_channels=4))

    assert len(model.encoder.convs) == 3
    assert model.encoder.convs[-1].out_channels == 4
