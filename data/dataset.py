"""
Dataset definitions.
Author: JiaWei Jiang

This file contains definitions of multiple datasets used in different
scenarios.

* [ ] Add raw runtime as aux task (e.g., log runtime w/o norm).
"""
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data.data import BaseData, Data

from metadata import NODE_CONFIG_FEAT_DIM
from paths import PROC_DATA_PATH

EPS = 1e-6


class TileDataset(Dataset):
    """Dataset for the collection `tile-xla`.

    Parameters:
        data: processed data
    """

    def __init__(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ):
        self.data = data
        self.target_col = kwargs["target_col"]

        self._n_samples = len(data)

        if np.sum(data["config_runtime"].apply(lambda x: np.sum(x))) == 0:
            self._infer = True
        else:
            self._infer = False
            self._add_target_to_data()

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        data_row = self.data.iloc[idx]
        data_sample = {
            "node_feat": torch.tensor(data_row["node_feat"], dtype=torch.float32),
            "node_opcode": torch.tensor(data_row["node_opcode"], dtype=torch.int32),
            "edge_index": torch.tensor(data_row["edge_index"], dtype=torch.long),
            "config_feat": torch.tensor(data_row["config_feat"], dtype=torch.float32),
        }

        # Add target
        if not self._infer:
            data_sample["target"] = torch.tensor(data_row[self.target_col], dtype=torch.float32)

        return data_sample

    def _add_target_to_data(self) -> None:
        target = []
        target_norm = []
        for i, r in self.data.iterrows():
            runtime = r["config_runtime"] / (r["config_runtime_normalizers"] + EPS)
            target.append(runtime)

            # Normalize
            runtime_norm = (runtime - np.mean(runtime)) / (np.std(runtime) + EPS)
            target_norm.append(runtime_norm)

        self.data["target"] = target
        self.data["target_norm"] = target_norm


class LayoutDataset(Dataset):
    """Layout dataset.

    Parameters:
        coll: data collection with the pattern source-search-split
        max_seg_size: maximum number of nodes per gragh segment
    """

    LAYOUT_ROOT: str = os.path.join(PROC_DATA_PATH, "layout")

    def __init__(self, data: pd.DataFrame, coll: str, max_seg_size: int = 1000, **kwargs: Dict[str, Any]) -> None:
        super().__init__()

        self.data = data
        self.src, self.search, self.split = coll.split("-")
        self.max_seg_size = max_seg_size
        self.data_root = os.path.join(self.LAYOUT_ROOT, f"{self.src}/{self.search}/")

        split = "test" if self.split == "test" else "train"
        self.node_config_feat_root = os.path.join(self.data_root, f"node_config_feat/{split}")
        self.seg_ptr_map, self.n_segs_map, self.hash_head_map, self.tot_n_segs = self._pre_segment()

    def _pre_segment(self) -> Tuple[Dict[int, Tensor], Dict[int, int], Dict[int, int], int]:
        seg_ptr_map, n_segs_map, hash_head_map = {}, {}, {}
        n_segs_accum = 0
        for i, data_row in self.data.iterrows():
            n_nodes = data_row["n_nodes"]
            seg_ptr = self._segment(n_nodes)
            n_segs = len(seg_ptr) - 1
            seg_ptr_map[i] = seg_ptr
            n_segs_map[i] = torch.tensor(n_segs, dtype=torch.int32)
            hash_head_map[i] = torch.tensor(n_segs_accum, dtype=torch.int32)
            n_segs_accum += n_segs

        return seg_ptr_map, n_segs_map, hash_head_map, n_segs_accum

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> BaseData:
        data_row = self.data.iloc[idx]
        node_config_feat = np.load(os.path.join(self.node_config_feat_root, data_row["file"]))["node_config_feat"]

        # Parse data fields
        node_feat = torch.tensor(data_row["node_feat"], dtype=torch.float32)
        node_opcode = torch.tensor(data_row["node_opcode"], dtype=torch.int32)
        edge_index = torch.tensor(data_row["edge_index"].T, dtype=torch.long)
        node_config_feat = torch.tensor(node_config_feat, dtype=torch.float32)
        node_config_ids = torch.tensor(data_row["node_config_ids"])

        # Derive simple graph stats
        n_nodes = torch.tensor(node_feat.shape[0], dtype=torch.int32)
        n_edges = torch.tensor(edge_index.shape[1], dtype=torch.int32)
        n_configs = torch.tensor(node_config_feat.shape[0], dtype=torch.int32)
        n_config_nodes = torch.tensor(node_config_feat.shape[1], dtype=torch.int32)

        data_sample = Data(
            node_feat=node_feat,  # (n, 140)
            node_opcode=node_opcode,  # (n, )
            edge_index=edge_index,  # (2, m)
            node_config_feat=node_config_feat.view(-1, NODE_CONFIG_FEAT_DIM),  # (c * nc, 18)
            node_config_ids=node_config_ids,  # (nc, )
            n_nodes=n_nodes,  # n
            n_edges=n_edges,  # m
            n_configs=n_configs,  # c
            n_config_nodes=n_config_nodes,  # nc
            seg_ptr=self.seg_ptr_map[idx],
            n_segs=self.n_segs_map[idx],
            hash_head=self.hash_head_map[idx],  # torch.tensor(idx, dtype=torch.int32),
        )

        if self.split != "test":
            runtime = torch.tensor(data_row["config_runtime"])  # (c, )
            data_sample.y = np.log1p(runtime)

        return data_sample

    def _segment(self, n_nodes: int) -> Tensor:
        """Segment the graph.

        Parameters:
            n_nodes: number of nodes in the graph

        Return:
            seg_ptr: graph segment break points, including the head
                and the tail
        """
        n_segs = n_nodes // self.max_seg_size + 1
        n_nodes_per_seg = n_nodes // n_segs
        seg_ptr = torch.arange(0, n_nodes, n_nodes_per_seg + 1)
        if seg_ptr[-1] != n_nodes:
            # Include all the rest nodes
            seg_ptr = torch.cat([seg_ptr, torch.tensor([n_nodes])])

        return seg_ptr
