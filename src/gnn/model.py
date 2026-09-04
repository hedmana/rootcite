"""The link-prediction model: encode every paper, then score a citation between two.

The encoder is deliberately direction-blind and the decoder is not. A paper is
described as much by what cites it as by what it cites, so message passing runs
over the undirected view of the graph. Whether a citation is plausible is a
different question: it has a direction, and a symmetric dot product cannot tell
a citation from its reverse. The decoder therefore projects the citing end and
the cited end through separate weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch_geometric.nn import SAGEConv


@dataclass
class ModelConfig:
    hidden_channels: int = 64
    out_channels: int = 32
    layers: int = 2
    dropout: float = 0.3


class CitationEncoder(nn.Module):
    """GraphSAGE over the citation neighbourhood."""

    def __init__(self, in_channels: int, config: ModelConfig) -> None:
        super().__init__()
        # The dataset ships raw log counts, whose scales differ by an order of
        # magnitude; without this the degree features drown the rest.
        self.normalize = nn.BatchNorm1d(in_channels)
        widths = (
            [in_channels] + [config.hidden_channels] * (config.layers - 1) + [config.out_channels]
        )
        self.convs = nn.ModuleList(SAGEConv(*width) for width in pairwise(widths))
        self.dropout = config.dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.normalize(x)
        for conv in self.convs[:-1]:
            activated = torch.relu(conv(x, edge_index))
            x = functional.dropout(activated, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class CitationDecoder(nn.Module):
    """Score an ordered pair as a citation the citing paper would make."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.citing = nn.Linear(channels, channels, bias=False)
        self.cited = nn.Linear(channels, channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        source, target = edge_label_index
        return (self.citing(z[source]) * self.cited(z[target])).sum(dim=-1) + self.bias


class LinkPredictor(nn.Module):
    """Encoder and decoder as one module, so a checkpoint is one object."""

    def __init__(self, in_channels: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = CitationEncoder(in_channels, self.config)
        self.decoder = CitationDecoder(self.config.out_channels)

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        return self.decoder(z, edge_label_index)

    def forward(self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor) -> Tensor:
        return self.decode(self.encode(x, edge_index), edge_label_index)
