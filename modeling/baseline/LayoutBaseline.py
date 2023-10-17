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

from metadata import N_OPS

# from modeling.layers import _GNNLayer


class LayoutNet(nn.Module):
    def __init__(
        self,
        op_emb_dim: int = 32,
        n_layers: int = 3,
        gconv_type: str = "SAGEConv",
        h_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        self.name = self.__class__.__name__
        super(LayoutNet, self).__init__()

        # Network parameters
        cat_dim = 140 + op_emb_dim + 18  # 190
        out_dim = 1  # Output one runtime scaler for ranking

        # Model blocks
        self.op_emb = nn.Embedding(N_OPS, op_emb_dim)
        self.lin = nn.Sequential(nn.Linear(cat_dim, 128), nn.ReLU())
        self.gnn_layers = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = 128 if layer == 0 else h_dim
            self.gnn_layers.append(SAGEConv(in_dim, h_dim, normalize=True, project=True))
        self.layer_post_mp = nn.Linear(h_dim, out_dim)
        self.sed = StaleEmbDropout()

    def forward(
        self, inputs: Batch, inputs_other: Optional[List[Tensor]] = None, batch_n_segs: Optional[List[int]] = None
    ) -> Tensor:
        x_node_feat = inputs.node_feat  # (BN, 140)
        x_config_feat = inputs.node_config_feat  # (BN, 18)

        # Op-code embedding
        x_op = self.op_emb(inputs.node_opcode)  # (BN, op_emb_dim)

        # Fuse node-level features
        x = torch.cat([x_node_feat, x_op, x_config_feat], dim=-1)
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
        if inputs_other is not None:
            output = self.sed(inputs_other, batch_n_segs, x_graph)
        else:
            # All segments go through forward pass
            output = None

        return output, x_graph


class StaleEmbDropout(nn.Module):
    """Stale Embedding Dropout (SED).

    Drop only stale segment embeddings.
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super(StaleEmbDropout, self).__init__()

        self.dropout = dropout

    def forward(self, inputs: List[Tensor], batch_n_segs: List[int], x_graph: Tensor) -> Tensor:
        """Forward pass.

        Shape:
            inputs: n_graphs_in_batch >> n_segs_per_graph (diff) >> n_configs_to_sample (32)
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
        x_graph_other = torch.zeros_like(x_graph)  # (BC, 1)
        i = 0  # Outer i to avoid graph having only one seg
        for n_segs in batch_n_segs:
            for j in range(n_segs - 1):
                # For all remaining segments
                # Currently sample only one segment to train
                x_graph_other[i, :] = x_graph_other[i, :] + inputs[i + 32 * j, :]
            if n_segs != 1:
                i += 1
        batch_n_segs = torch.Tensor(batch_n_segs).view(-1, 1).to(inputs.device)
        eta = self.dropout + (1 - self.dropout) * batch_n_segs
        output = x_graph * eta + x_graph_other

        return output


class HistoryEmbTable(torch.nn.Module):
    """Historical Embedding Table."""

    def __init__(self, num_embeddings: int = int(6e7), embedding_dim: int = 1, device: str = "cuda:0") -> None:
        super(HistoryEmbTable, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._device = torch.device(device)

        pin_memory = device == "cpu"
        self.emb = torch.empty(
            num_embeddings, embedding_dim, dtype=torch.float32, device=device, pin_memory=pin_memory
        )
        self.emb.fill_(0.0)

    @torch.no_grad()
    def pull(self, hash_idx: int) -> Tensor:
        x_emb = self.emb[hash_idx]

        return x_emb

    @torch.no_grad()
    def push(
        self,
        x: Tensor,
        hash_idx: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        count: Optional[Tensor] = None,
    ) -> None:
        if hash_idx is None and x.size(0) != self.num_embeddings:
            raise ValueError
        elif hash_idx is None and x.size(0) == self.num_embeddings:
            self.emb.copy_(x)
        elif offset is None or count is None:
            self.emb[hash_idx] = x
        else:
            # Push chunk-by-chunk
            src_o = 0
            x = x.to(self.emb.device)
            for (
                dst_o,
                c,
            ) in zip(offset.tolist(), count.tolist()):
                self.emb[dst_o : dst_o + c] = x[src_o : src_o + c]
                src_o += c

    def forward(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    # def _apply(self, fn):
    # Set the `_device` of the module without transfering `self.emb`.
    # self._device = fn(torch.zeros(1)).device
    # return self
