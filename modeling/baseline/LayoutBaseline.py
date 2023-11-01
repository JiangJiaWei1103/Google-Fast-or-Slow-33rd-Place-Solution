"""
Baseline model architectures for layout optimization.
Author: JiaWei Jiang

This file contains definition of simple baseline model architectures.
Please see https://github.com/kaidic/GST.

Commonly used notations are defined as follows:
* `B`: batch size
* `N`: #nodes
* `M`: #edges
* `C`: #configurations for a graph
* `BN`: #nodes in one batch
* `BM`: #edges in one batch
* `BC`: #configurations in one batch
"""
from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.pool import global_max_pool, global_mean_pool

from metadata import N_OPS, NODE_CONFIG_FEAT_DIM


class LayoutEarlyJoinGConv(nn.Module):
    """Graph convolutional model with early-join config features.

    For layouts, config features are at the node-level; hence, there's
    no way to implement late-join model.

    Parameters:
        node_feat_dim: node feature dimension, excluding the last for
            shape element type embedding
        n_shape_ele_types: number of shape element types
        shape_ele_type_emb_dim: shape element type embedding dimension
        op_emb_dim: op-code embedding dimension
        n_layers: number of GNN layers
        gconv_type: type of graph convolution
        h_dim: hidden dimension
        dropout: dropout ratio
        sed_dropout: dropout ratio of StaleEmbDropout
    """

    def __init__(
        self,
        node_feat_dim: int,
        n_shape_ele_types: int,
        shape_ele_type_emb_dim: int = 4,
        op_emb_dim: int = 32,
        n_layers: int = 3,
        gconv_type: str = "SAGEConv",
        h_dim: int = 64,
        dropout: Optional[float] = None,
        sed_dropout: Optional[float] = None,
    ) -> None:
        super(LayoutEarlyJoinGConv, self).__init__()

        # Network parameters
        cat_dim = node_feat_dim + shape_ele_type_emb_dim + op_emb_dim + NODE_CONFIG_FEAT_DIM
        out_dim = 1

        # Model blocks
        self.op_emb = nn.Embedding(N_OPS, op_emb_dim)
        self.shape_ele_type_emb = nn.Embedding(n_shape_ele_types, shape_ele_type_emb_dim)
        self.lin = nn.Sequential(nn.Linear(cat_dim, h_dim * 2), nn.ReLU())
        self.gnn_layers = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = h_dim * 2 if layer == 0 else h_dim
            self.gnn_layers.append(SAGEConv(in_dim, h_dim, normalize=True, project=True))
        self.layer_post_mp = nn.Linear(h_dim, out_dim)
        if sed_dropout is not None:
            self.sed = StaleEmbDropout(sed_dropout)
        else:
            self.sed = None

    def forward(
        self, inputs: Batch, inputs_other: Optional[List[Tensor]] = None, batch_n_segs: Optional[List[int]] = None
    ) -> Tensor:
        x_node_feat = inputs.node_feat  # (BN, node_feat_dim+1)
        x_config_feat = inputs.node_config_feat  # (BN, NODE_CONFIG_FEAT_DIM)

        # Categorical embedding
        x_shape_ele_type = self.shape_ele_type_emb(x_node_feat[:, -1].long())  # (BN, shape_ele_type_emb_dim)
        x_op = self.op_emb(inputs.node_opcode)  # (BN, op_emb_dim)

        # Fuse node-level features
        x = torch.cat([x_node_feat[:, :-1], x_shape_ele_type, x_op, x_config_feat], dim=-1)
        x = self.lin(x)
        inputs.x = x

        # GNN layers
        for _, gnn_layer in enumerate(self.gnn_layers):
            inputs.x = gnn_layer(inputs.x, inputs.edge_index)  # (BN, h_dim)

        # Pool to sub-graph context
        x_graph = global_max_pool(inputs.x, inputs.batch) + global_mean_pool(inputs.x, inputs.batch)
        x_graph = x_graph / torch.norm(x_graph, dim=-1, keepdim=True)
        x_graph = self.layer_post_mp(x_graph)  # (BC, out_dim), out_dim=1

        # Apply SED and get the global graph context
        output = None
        if self.training:
            assert inputs_other is not None, "History embeddings must be provided for dropout."
            if self.sed is not None and len(inputs_other) > 0:
                output = self.sed(inputs_other, batch_n_segs, x_graph)
            else:
                # One seg for whole graph (scale the prediction), also cover n_segs == 1
                batch_n_segs = torch.Tensor(batch_n_segs).view(-1, 1).to(x_graph.device)
                output = x_graph * batch_n_segs
        else:
            # Only one segment per graph in a batch,
            # or sed is disabled
            output = x_graph

        return output, x_graph


class StaleEmbDropout(nn.Module):
    """Stale Embedding Dropout (SED).

    Drop only stale segment embeddings.
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super(StaleEmbDropout, self).__init__()

        # p for Bernoulli
        self.dropout = 1 - dropout

    def forward(self, inputs: List[Tensor], batch_n_segs: List[int], x_graph: Tensor) -> Tensor:
        """Forward pass.

        Shape:
            inputs: flatten (B, n_segs_per_graph, n_configs_to_sample)
                *Note: n_segs_per_graph is actually (n_segs - 1), one
                    segment for training
            batch_n_segs: (BC, )
            x_graph: (BC, out_dim), where C denotes the fixed number,
                n_configs_to_sample for each graph
        """
        inputs = torch.stack(inputs)
        mask = torch.zeros_like(inputs, device=inputs.device)
        mask.fill_(self.dropout)
        mask = torch.bernoulli(mask)
        inputs = inputs * mask

        # Pool to sub-graph context
        batch_n_segs = torch.tensor(batch_n_segs).to(x_graph.device)
        inputs = inputs.reshape(-1, 200)
        batch_n_others = batch_n_segs[::200] - 1  # Minus the trained seg
        have_others = batch_n_others != 0
        batch_n_others = batch_n_others[have_others]
        pool_idx = torch.repeat_interleave(torch.arange(len(batch_n_others)).to(x_graph.device), batch_n_others).to(
            inputs.device
        )
        pool_idx = pool_idx.reshape(-1, 1).expand(-1, 200)
        x_graph_other = torch.zeros(len(batch_n_others), 200, dtype=torch.float32, device=inputs.device)
        x_graph_other = x_graph_other.scatter_add_(0, pool_idx, inputs)

        x_graph = x_graph.reshape(-1, 200)
        eta = self.dropout + (1 - self.dropout) * (batch_n_others + 1)
        output = x_graph * eta.reshape(-1, 1)
        output[have_others, :] = output[have_others, :] + x_graph_other

        return output


class HistoryEmbTable(torch.nn.Module):
    """Historical Embedding Table."""

    MAX_N_CONFIGS: int = 100040

    def __init__(self, tot_n_segs: int, device: str = "cuda:0") -> None:
        super(HistoryEmbTable, self).__init__()

        self.tot_n_segs = tot_n_segs
        self.table_size = int(tot_n_segs * self.MAX_N_CONFIGS)
        self._device = torch.device(device)

        pin_memory = device == "cpu"
        self.emb = torch.empty(
            tot_n_segs, self.MAX_N_CONFIGS, dtype=torch.float32, device=device, pin_memory=pin_memory
        )
        self.emb.fill_(0.0)

    @torch.no_grad()
    def pull(self, hash_idx: List[int]) -> Tensor:
        ij, k = hash_idx
        x_emb = self.emb[ij][k]

        return x_emb

    @torch.no_grad()
    def push(self, x: Tensor, hash_idx: List[int]) -> None:
        ij, k = hash_idx
        self.emb[ij][k] = x

    def forward(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError
