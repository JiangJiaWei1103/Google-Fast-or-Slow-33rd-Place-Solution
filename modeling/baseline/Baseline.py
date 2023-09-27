"""
Baseline model architectures.
Author: JiaWei Jiang

This file contains definition of simple baseline model architectures.
Please see https://github.com/google-research-datasets/tpu_graphs.

Commonly used notations are defined as follows:
* `B`: batch size
* `N`: #nodes
* `M`: #edges
* `C`: #configurations for a graph
* `BN`: #nodes in one batch
* `BM`: #edges in one batch
* `BC`: #configurations in one batch
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.pool import global_add_pool

# from .common import Mish, Swish
from metadata import N_OPS


class LateJoinSAGE(nn.Module):
    """GraphSAGE model with late-join config features.

    Parameters:
        op_emb_dim: op-code embedding dimension
        n_layers: number of GNN layers

    """

    def __init__(
        self,
        op_emb_dim: int = 8,
        n_layers: int = 3,
        gconv_type: str = "SAGEConv",
        h_dim: int = 64,
        act: str = "leaky_relu",
        dropout: float = 0.2,
        ln: bool = False,
    ) -> None:
        super(LateJoinSAGE, self).__init__()

        # Network parameters

        # Model blocks
        self.op_emb = nn.Embedding(N_OPS, op_emb_dim)
        self.gnn_layers = nn.ModuleList()
        for layer in range(n_layers):
            if layer == 0:
                self.gnn_layers.append(_GNNLayer(in_dim=148))
            else:
                self.gnn_layers.append(_GNNLayer(in_dim=h_dim))
        self.postnet = nn.Sequential(nn.Linear(h_dim + 24, h_dim), nn.LeakyReLU(inplace=True), nn.Linear(h_dim, 1))

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Forward pass.

        Shape:
            node_feat: (BN, 140)
            node_opcode: (BN)
            edge_index: (BM, 2), M can be different
                *Note: Add graph collate
            config_feat: (BC, 24)
        """
        node_feat = inputs["node_feat"]
        node_opcode = inputs["node_opcode"]
        edge_index = inputs["edge_index"]
        config_feat = inputs["config_feat"]
        n_configs = inputs["n_configs"]

        # Op-code embedding
        x_op = self.op_emb(node_opcode)  # (BN, op_emb_dim)

        # Graph convolution layers
        x = torch.cat([node_feat, x_op], dim=-1)
        for layer, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)  # (BN, h_dim)

        # Pool nodes to graph context
        x = global_add_pool(x, inputs["batch"])  # (B, h_dim)

        # Late-join config features
        out_size = torch.sum(n_configs).item()
        x = torch.repeat_interleave(x, n_configs, 0, output_size=out_size)
        x = torch.cat([x, config_feat], dim=-1)  # (BC, h_dim + 24)

        # Output layer
        output = self.postnet(x).squeeze(dim=-1)  # (BC)

        return output


class _GNNLayer(nn.Module):
    """GNN layer.

    Parameters:
        gconv_type: type of graph convolution
    """

    def __init__(
        self,
        gconv_type: str = "SAGEConv",
        in_dim: int = 128,
        h_dim: int = 64,
        act: str = "leaky_relu",
        dropout: Optional[float] = None,
        ln: bool = False,
    ) -> None:
        super(_GNNLayer, self).__init__()

        # Network parameters
        self.gconv_type = gconv_type
        self.h_dim = h_dim

        # Model blocks
        self.gconv = _BidirGConv(gconv_type, in_dim, in_dim)
        self.linear = nn.Linear(in_dim * 2, h_dim)
        if act == "leaky_relu":
            self.act = nn.LeakyReLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if ln:
            self.layer_norm = nn.LayerNorm(h_dim)
        else:
            self.layer_norm = None

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x_conv = self.gconv(x, edge_index)
        x = torch.cat([x_conv, x], axis=-1)
        x = self.linear(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x


class _BidirGConv(nn.Module):
    """Bi-directional graph convolution with directed graphs.

    The concept can be implemented by doing graph conv over undirected
    graphs.

    Parameters:

    """

    def __init__(
        self,
        gconv_type: str = "SAGEConv",
        in_dim: int = 128,
        h_dim: int = 128,
    ) -> None:
        super(_BidirGConv, self).__init__()

        # Network parameters
        self.gconv_type = gconv_type

        # Model blocks
        if gconv_type == "SAGEConv":
            self.gconv = SAGEConv(in_dim, h_dim, normalize=True)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Construct the undirected graph
        edge_index_t = edge_index[..., [1, 0]]
        edge_index_bidir = torch.cat([edge_index, edge_index_t], dim=0)

        # Graph convolution
        x = self.gconv(x, edge_index_bidir.T)

        return x
