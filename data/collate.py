"""
Customized collate functions.
Author: JiaWei Jiang
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


def tile_collate(batch: List[Any], *, collate_fn_map: Optional[Any] = None) -> Dict[str, Tensor]:
    """Collate function of collection tile.

    This collate is used to deal with the unaligned number of nodes and
    configurations of different kernels.
    """
    n_nodes_accum = 0
    n_configs_flat = []
    batch_vec = []
    batch_collated = defaultdict(list)
    for i, d in enumerate(batch):
        n_nodes, n_configs = len(d["node_feat"]), len(d["config_feat"])  # N, C
        batch_vec.append(torch.tensor([i for _ in range(n_nodes)], dtype=torch.int32))
        n_configs_flat.append(n_configs)

        edge_index_incre = d["edge_index"]
        edge_index_incre += n_nodes_accum
        n_nodes_accum += n_nodes

        for field in ["node_feat", "node_opcode", "config_feat", "target"]:
            if field in d:
                batch_collated[field].append(d[field])
        batch_collated["edge_index"].append(edge_index_incre)

    # List to tensor
    for field, dtype in {
        "node_feat": torch.float32,
        "node_opcode": torch.int32,
        "edge_index": torch.long,
        "config_feat": torch.float32,
        "target": torch.float32,
    }.items():
        if field in batch_collated:
            batch_collated[field] = torch.cat(batch_collated[field], dim=0).to(dtype=dtype)
    batch_collated["batch"] = torch.cat(batch_vec, dim=-1).long()
    batch_collated["n_configs"] = torch.tensor(n_configs_flat, dtype=torch.int32)

    return batch_collated
