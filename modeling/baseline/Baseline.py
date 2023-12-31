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
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool

from metadata import CONFIG_FEAT_DIM, N_OPS, NODE_FEAT_DIM
from modeling.layers import _GNNLayer


class TileEarlyJoinGConv(nn.Module):
    """Graph convolutional model with early-join config features.

    For tile, config feature is at the graph-level, which will be
    broadcasted to each node.

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
        node_feat_dim: int = 85,
        n_shape_ele_types: int = 8,
        shape_ele_type_emb_dim: int = 4,
        op_emb_dim: int = 32,
        n_layers: int = 3,
        gconv_type: str = "SAGEConv",
        h_dim: int = 64,
        dropout: Optional[float] = None,
        sed_dropout: Optional[float] = None,
    ) -> None:
        super(TileEarlyJoinGConv, self).__init__()

        # Network parameters
        cat_dim = node_feat_dim + shape_ele_type_emb_dim + op_emb_dim + CONFIG_FEAT_DIM
        out_dim = 1

        # Model blocks
        self.op_emb = nn.Embedding(N_OPS, op_emb_dim)
        self.shape_ele_type_emb = nn.Embedding(n_shape_ele_types, shape_ele_type_emb_dim)
        self.lin = nn.Sequential(nn.Linear(cat_dim, h_dim * 2), nn.ReLU(), nn.Linear(h_dim * 2, h_dim * 2), nn.ReLU())
        self.gnn_layers = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = h_dim * 2 if layer == 0 else h_dim
            if gconv_type == "SAGEConv":
                gnn_layer = _GNNLayer(
                    gconv_type="SAGEConv2D",
                    in_dim=in_dim,
                    h_dim=h_dim,
                    bidir=False,
                )
            self.gnn_layers.append(gnn_layer)
        self.layer_post_mp = nn.Sequential(nn.Linear(h_dim, h_dim // 2), nn.ReLU(), nn.Linear(h_dim // 2, out_dim))

    def forward(
        self, inputs: Batch, inputs_other: Optional[List[Tensor]] = None, batch_n_segs: Optional[List[int]] = None
    ) -> Tensor:
        x_node_feat = inputs.node_feat  # (BN, node_feat_dim+1)
        x_config_feat = inputs.config_feat.reshape(-1, 1000, CONFIG_FEAT_DIM)  # (B, C, CONFIG_FEAT_DIM)
        n_nodes = x_node_feat.shape[0]
        batch_size, n_configs, _ = x_config_feat.shape

        # Categorical embedding
        x_shape_ele_type = self.shape_ele_type_emb(x_node_feat[:, -1].long())  # (BN, shape_ele_type_emb_dim)
        x_op = self.op_emb(inputs.node_opcode)  # (BN, op_emb_dim)

        # Fuse node-level features
        x_node = torch.cat(
            [x_node_feat[:, :-1], x_shape_ele_type, x_op], dim=-1
        )  # (BN, node_feat_dim+CONFIG_FEAT_DIM+shape_ele_type_emb_dim)
        x_node = x_node.unsqueeze(dim=1).expand(-1, n_configs, -1)  # (BN, C, ...)
        x_config_feat = torch.repeat_interleave(x_config_feat, inputs.n_nodes, dim=0)  # (BN, C, NODE_CONFIG_FEAT_DIM)
        x = torch.cat([x_node, x_config_feat], dim=-1)  # (BN, C, cat_dim)
        x = self.lin(x)  # (BN, C, h_dim * 2)
        inputs.x = x

        # GNN layers
        for _, gnn_layer in enumerate(self.gnn_layers):
            h = gnn_layer(inputs.x, inputs.edge_index)  # (BN, C, h_dim)
            inputs.x = h

        # Pool to sub-graph context
        x = inputs.x.reshape(n_nodes, -1)
        x_graph = global_max_pool(x, inputs.batch) + global_mean_pool(x, inputs.batch)  # (B, C * h_dim)
        x_graph = x_graph.reshape(batch_size, n_configs, -1)
        x_graph = x_graph / torch.norm(x_graph, dim=-1, keepdim=True)  # (B, C, h_dim)

        # Prediction head
        output = self.layer_post_mp(x_graph).squeeze(dim=-1)  # (B, C)

        return output


class EarlyJoinGConv(nn.Module):
    """Graph convolutional model with early-join config features.

    Parameters:
        op_emb_dim: op-code embedding dimension
        n_layers: number of GNN layers
        gconv_type: type of graph convolution
        h_dim: hidden dimension
        act: activation function
        dropout: dropout ratio
        ln: if True, layernorm is applied
    """

    def __init__(
        self,
        op_emb_dim: int = 8,
        n_layers: int = 3,
        gconv_type: str = "SAGEConv2D",
        h_dim: int = 64,
        dropout: float = 0.2,
        ln: bool = False,
    ) -> None:
        super(EarlyJoinGConv, self).__init__()

        # Network parameters
        gnn_in_dim = NODE_FEAT_DIM + op_emb_dim + CONFIG_FEAT_DIM

        # Model blocks
        self.op_emb = nn.Embedding(N_OPS, op_emb_dim)
        self.gnn_layers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = gnn_in_dim if layer == 0 else h_dim
            gnn_layer = _GNNLayer(
                gconv_type=gconv_type,
                in_dim=in_dim,
                h_dim=h_dim,
                bidir=True,
            )
            self.gnn_layers.append(gnn_layer)
            self.skip_projs.append(nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)))
        self.postnet = nn.Sequential(
            nn.Linear(n_layers * h_dim, (n_layers - 1) * h_dim),
            nn.ReLU(),
            nn.Linear((n_layers - 1) * h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Forward pass.

        Shape:
            node_feat: (1, N, NODE_FEAT_DIM)
            node_opcode: (1, N)
            edge_index: (1, M, 2), M can be different
            config_feat: (1, C, CONFIG_FEAT_DIM), C is the max number
                of configurations in current batch
        """
        node_feat = inputs["node_feat"].squeeze(dim=0)
        node_opcode = inputs["node_opcode"].squeeze(dim=0)
        edge_index = inputs["edge_index"].squeeze(dim=0)
        config_feat = inputs["config_feat"].squeeze(dim=0)
        n_nodes, n_configs = node_feat.shape[0], config_feat.shape[0]

        # Op-code embedding
        x_op = self.op_emb(node_opcode)  # (N, op_emb_dim)

        # Broadcast config_feat to each node in early-join manner
        x = torch.cat([node_feat, x_op], dim=-1)
        x = x.unsqueeze(1).expand(-1, n_configs, -1)
        config_feat = config_feat.unsqueeze(0).expand(n_nodes, -1, -1)
        x = torch.cat([x, config_feat], dim=-1)  # (N, C, gnn_in_dim)

        # Graph convolution layers
        x_skip = []
        for layer, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)  # (N, C, h_dim)

            x_skip.append(self.skip_projs[layer](x))

        # Pool nodes to graph context
        x = torch.cat(x_skip, dim=-1)  # (N, C, n_layers * h_dim)
        x = x.reshape(n_nodes, -1)
        x = global_add_pool(x, torch.zeros(n_nodes, dtype=torch.long, device=x.device))

        # Output layer
        x = x.reshape(n_configs, -1)  # (C, n_layers * h_dim)
        output = self.postnet(x).squeeze(dim=-1)  # (C, )

        return output


class LateJoinGConv(nn.Module):
    """Graph convolutional model with late-join config features.

    Parameters:
        op_emb_dim: op-code embedding dimension
        n_layers: number of GNN layers
        gconv_type: type of graph convolution
        h_dim: hidden dimension
        dropout: dropout ratio
        ln: if True, layernorm is applied
    """

    def __init__(
        self,
        op_emb_dim: int = 8,
        n_layers: int = 3,
        gconv_type: str = "SAGEConv1D",
        h_dim: int = 64,
        dropout: float = 0.2,
        ln: bool = False,
        bidir: bool = True,
    ) -> None:
        super(LateJoinGConv, self).__init__()

        # Network parameters
        shape_ele_type_emb_dim = 4
        gnn_in_dim = (NODE_FEAT_DIM - 1) + op_emb_dim + shape_ele_type_emb_dim

        # Model blocks
        self.op_emb = nn.Embedding(N_OPS, op_emb_dim)
        self.shape_ele_type_emb = nn.Embedding(8, shape_ele_type_emb_dim)
        self.gnn_layers = nn.ModuleList()
        # self.skip_projs = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = gnn_in_dim if layer == 0 else h_dim
            gnn_layer = _GNNLayer(
                gconv_type=gconv_type,
                in_dim=in_dim,
                h_dim=h_dim,
                bidir=bidir,
            )
            self.gnn_layers.append(gnn_layer)
        # self.skip_projs.append(nn.Sequential(
        #     nn.Linear(h_dim, h_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # ))
        self.dropout = nn.Dropout(dropout)
        self.postnet = nn.Sequential(nn.Linear(2 * h_dim + CONFIG_FEAT_DIM, h_dim), nn.ReLU(), nn.Linear(h_dim, 1))

    # self.postnet = nn.Sequential(
    #     nn.Linear(n_layers * h_dim + CONFIG_FEAT_DIM, (n_layers-1) * h_dim),
    #     nn.ReLU(),
    #     nn.Linear((n_layers-1) * h_dim, h_dim),
    #     nn.ReLU(),
    #     nn.Linear(h_dim, 1)
    # )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Forward pass.

        Shape:
            node_feat: (BN, NODE_FEAT_DIM)
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
        x_shape_ele_type = self.shape_ele_type_emb(node_feat[:, -1].int())

        # Graph convolution layers
        x = torch.cat([node_feat[:, :-1], x_op, x_shape_ele_type], dim=-1)
        # x_skip = []
        for layer, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)  # (BN, h_dim)

            x = self.dropout(x)

        #   x_skip.append(self.skip_projs[layer](x))

        # Pool nodes to graph context
        # x = torch.cat(x_skip, dim=-1)  # (BN, n_layers * h_dim)
        # x = global_add_pool(x, inputs["batch"])  # (B, n_layers * h_dim)

        x_avg = global_mean_pool(x, inputs["batch"])
        x_max = global_max_pool(x, inputs["batch"])
        x = torch.cat([x_avg, x_max], dim=-1)

        # Late-join config features
        out_size = torch.sum(n_configs).item()
        x = torch.repeat_interleave(x, n_configs, 0, output_size=out_size)
        x = torch.cat([x, config_feat], dim=-1)  # (BC, n_layers * h_dim + 24)

        # Output layer
        output = self.postnet(x).squeeze(dim=-1)  # (BC)

        return output


class LateJoinGAT(nn.Module):
    """Graph convolutional model with late-join config features.

    A simpler model architeture

    Parameters:
        op_emb_dim: op-code embedding dimension
        n_layers: number of GNN layers
        gconv_type: type of graph convolution
        h_dim: hidden dimension
        dropout: dropout ratio
    """

    def __init__(
        self,
        op_emb_dim: int = 8,
        n_layers: int = 3,
        gconv_type: str = "GATConv",
        n_heads: int = 4,
        h_dim: int = 64,
        dropout: float = 0.2,
        bidir: bool = True,
    ) -> None:
        super(LateJoinGAT, self).__init__()

        # Network parameters
        shape_ele_type_emb_dim = 4
        gnn_in_dim = (NODE_FEAT_DIM - 1) + op_emb_dim + shape_ele_type_emb_dim

        # Model blocks
        self.op_emb = nn.Embedding(N_OPS, op_emb_dim)
        self.shape_ele_type_emb = nn.Embedding(8, shape_ele_type_emb_dim)
        self.gnn_layers = nn.ModuleList()
        self.act = nn.ELU()
        for layer in range(n_layers):
            in_dim = gnn_in_dim if layer == 0 else n_heads * h_dim
            gnn_layer = _GNNLayer(
                gconv_type=gconv_type,
                in_dim=in_dim,
                h_dim=h_dim,
                bidir=bidir,
                dropout=dropout,
                n_heads=n_heads,
            )
            self.gnn_layers.append(gnn_layer)
        self.postnet = nn.Sequential(
            nn.Linear(n_heads * h_dim + CONFIG_FEAT_DIM, h_dim), nn.ReLU(), nn.Linear(h_dim, 1)
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Forward pass.

        Shape:
            node_feat: (BN, NODE_FEAT_DIM)
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
        x_shape_ele_type = self.shape_ele_type_emb(node_feat[:, -1].int())

        # Graph convolution layers
        x = torch.cat([node_feat[:, :-1], x_op, x_shape_ele_type], dim=-1)
        for layer, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)  # (BN, n_heads * h_dim)

        # Pool nodes to graph context
        x = global_add_pool(x, inputs["batch"])  # (B, n_heads * h_dim)

        # Late-join config features
        out_size = torch.sum(n_configs).item()
        x = torch.repeat_interleave(x, n_configs, 0, output_size=out_size)
        x = torch.cat([x, config_feat], dim=-1)  # (BC, n_heads * h_dim + 24)

        # Output layer
        output = self.postnet(x).squeeze(dim=-1)  # (BC)

        return output
