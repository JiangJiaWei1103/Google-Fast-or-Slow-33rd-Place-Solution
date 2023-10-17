"""
Dataset definitions.
Author: JiaWei Jiang

This file contains definitions of multiple datasets used in different
scenarios.

* [ ] Add raw runtime as aux task (e.g., log runtime w/o norm).
"""
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data.data import BaseData, Data

from metadata import NODE_CONFIG_FEAT_DIM
from paths import RAW_DATA_PATH

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

    LAYOUT_ROOT: str = os.path.join(RAW_DATA_PATH, "npz_all/npz/layout")

    def __init__(self, coll: str, max_seg_size: int = 1000, **kwargs: Dict[str, Any]) -> None:
        super().__init__()

        self.src, self.search, self.split = coll.split("-")
        self.data_root = os.path.join(self.LAYOUT_ROOT, f"{self.src}/{self.search}/{self.split}")
        self._data_files = self._get_data_files()
        self.max_seg_size = max_seg_size

    def _get_data_files(self) -> List[str]:
        data_files = []
        for file in sorted(os.listdir(self.data_root)):
            if not file.endswith("npz"):
                continue
            data_files.append(file)

        return data_files

    def __len__(self) -> int:
        return len(self._data_files)

    def __getitem__(self, idx: int) -> BaseData:
        # Load data sample
        layout_tmp = dict(np.load(os.path.join(self.data_root, self._data_files[idx])))

        # Parse data fields
        node_feat = torch.tensor(layout_tmp["node_feat"], dtype=torch.float32)
        node_opcode = torch.tensor(layout_tmp["node_opcode"], dtype=torch.int32)
        edge_index = torch.tensor(layout_tmp["edge_index"].T, dtype=torch.long)
        node_config_feat = torch.tensor(layout_tmp["node_config_feat"], dtype=torch.float32).view(
            -1, NODE_CONFIG_FEAT_DIM
        )
        node_config_ids = torch.tensor(layout_tmp["node_config_ids"])

        # Derive simple graph stats
        n_nodes = torch.tensor(node_feat.shape[0], dtype=torch.int32)
        n_edges = torch.tensor(edge_index.shape[1], dtype=torch.int32)
        n_configs = torch.tensor(layout_tmp["node_config_feat"].shape[0], dtype=torch.int32)
        n_config_nodes = torch.tensor(layout_tmp["node_config_feat"].shape[1], dtype=torch.int32)

        # Segment the graph
        seg_ptr = self._segment(n_nodes)
        n_segs = torch.tensor(len(seg_ptr) - 1, dtype=torch.int32)
        if idx == 0:
            hash_head = torch.tensor(0, dtype=torch.int32)
        else:
            hash_head = torch.tensor(n_segs * n_configs + (idx - 1), dtype=torch.int32)

        data_sample = Data(
            node_feat=node_feat,  # (n, 140)
            node_opcode=node_opcode,  # (n, )
            edge_index=edge_index,  # (2, m)
            node_config_feat=node_config_feat,  # (c * nc, 18)
            node_config_ids=node_config_ids,  # (nc, )
            n_nodes=n_nodes,  # n
            n_edges=n_edges,  # m
            n_configs=n_configs,  # c
            n_config_nodes=n_config_nodes,  # nc
            seg_ptr=seg_ptr,
            n_segs=n_segs,
            hash_head=hash_head,
        )

        if self.split != "test":
            runtime = torch.tensor(layout_tmp["config_runtime"])  # (c, )
            data_sample.y = np.log1p(runtime)
            # data_sample.y = runtime
            # data_sample["target"]: runtime  # (c, )

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

        # from torch_geometric.data import Dataset
        # class LayoutDataset(Dataset):
        #    """Layout dataset.
        #
        #    Raw data is too large to fit into RAM.
        #    """
        #
        #    def __init__(
        #        self,
        #        root: str = "./data",
        #        src: str = "nlp",  # "nlp" or "xla"
        #        search: str = "random",  # "random" or "default"
        #        split: str = "train",  # "train", "valid" or "test"
        #        n_nodes_per_seg: int = 1000,
        #        transform: Optional[Callable] = None,
        #        pre_transform: Optional[Callable] = None,
        #        pre_filter: Optional[Callable] = None,
        #    ) -> None:
        #        self.src = src
        #        self.search = search
        #        self.split = split
        #        self.n_nodes_per_seg = n_nodes_per_seg
        #        self._n_samples = self._set_n_samples()
        #
        #        super().__init__(root, transform, pre_transform, pre_filter)
        #
        #    def _set_n_samples(self) -> int:
        #        data_root = f"./data/raw/npz_all/npz/layout/{self.src}/{self.search}/{self.split}"
        #        n_samples = 0
        #        for file in os.listdir(data_root):
        #            if not file.endswith("npz"): continue
        #            n_samples += 1
        #
        #        return n_samples
        #
        #    @property
        #    def raw_file_names(self) -> List[str]:
        #        """Return files needed to be found in `self.raw_dir` to skip
        #        download.
        #        """
        #        return [f"npz_all/npz/layout/{self.src}/{self.search}/{self.split}"]
        #
        #    @property
        #    def processed_file_names(self) -> List[str]:
        #        """Return files needed to be found in `self.processed_dir` to
        #        skip processing.
        #        """
        #        return [f"{self.src}-{self.search}-{self.split}_{i}.pt" for i in range(self._n_samples)]
        #
        #    def process(self) -> None:
        #        """Process the raw data and save it into the processed dir."""
        #        def _segment(n_nodes: int) -> Tensor:
        #            """Segment the graph.
        #
        #            Parameters:
        #                n_nodes: number of nodes in the graph
        #
        #            Return:
        #                seg_ptr: graph segment break points, including the head
        #                    and the tail
        #            """
        #            n_segs = n_nodes // self.n_nodes_per_seg + 1
        #            n_nodes_per_seg = n_nodes // n_segs
        #            seg_ptr = torch.arange(0, n_nodes, n_nodes_per_seg+1)
        #            if seg_ptr[-1] != n_nodes:
        #                # Include all the rest nodes
        #                seg_ptr = torch.cat([seg_ptr, torch.tensor([n_nodes])])
        #
        #            return seg_ptr
        #
        #        graph_cnt, seg_cnt = 0, 0
        #        for data_root in self.raw_paths:
        #            for file in os.listdir(data_root):
        #                if not file.endswith("npz"): continue
        #
        #                # Load graph data
        #                file_path = os.path.join(data_root, file)
        #                layout_tmp = dict(np.load(file_path))
        #                if "edge_index" not in layout_tmp:
        #                    logging.info(f"{file_path} has no edge_index.")
        #
        #                # Parse data fields
        #                node_feat = torch.tensor(layout_tmp["node_feat"])
        #                node_opcode = torch.tensor(layout_tmp["node_opcode"])
        #                edge_index = torch.tensor(layout_tmp["edge_index"].T)
        #                node_config_feat = torch.tensor(layout_tmp["node_config_feat"]).view(-1, CONFIG_FEAT_DIM)
        #                node_config_ids = torch.tensor(layout_tmp["node_config_ids"])
        #                runtime = torch.tensor(layout_tmp["config_runtime"])
        #
        #                # Derive simple graph stats
        #                n_nodes = torch.tensor(node_feat.shape[0])
        #                n_configs = torch.tensor(layout_tmp["node_config_feat"].shape[0])
        #                n_config_nodes = torch.tensor(layout_tmp["node_config_feat"].shape[1])
        #
        #                # Segment the graph
        #                seg_ptr = _segment(n_nodes)
        #
        #                # Build graph data object and dump it
        #                data = Data(
        #                    node_feat=node_feat,  # (n, 140)
        #                    node_opcode=node_opcode,  # (n, )
        #                    edge_index=edge_index,  # (2, m)
        #                    node_config_feat=node_config_feat,  # (c * nc, 18)
        #                    node_config_ids=node_config_ids,  # (nc, )
        #                    runtime=runtime,  # (c, )
        #                    n_nodes=n_nodes,  # n
        #                    n_configs=n_configs,  # c
        #                    n_config_nodes=n_config_nodes,  # nc
        #                    seg_ptr=seg_ptr,
        #                    # ===
        #                    partition_idx=seg_cnt
        #                    # ===
        #                )
        #                torch.save(data,
        # os.path.join(self.processed_dir, f"{self.src}-{self.search}-{self.split}_{graph_cnt}.pt"))
        #
        #                graph_cnt += 1
        #                # ===
        #                # Why * n_configs???
        #                n_segs = len(seg_ptr) - 1
        #                seg_cnt += n_segs * n_configs
        #                # ===
        #
        #    def len(self) -> int:
        #        return len(self.processed_paths)
        #
        #    def get(self, idx: int) -> Data:
        #        data_path = os.path.join(self.processed_dir, f"{self.src}-{self.search}-{self.split}_{idx}.pt")
        #        data_sample = torch.load(data_path)
        #
        #        return data_sample

        # class LayoutDatasetInMem(InMemoryDataset):
        #    """Layout dataset for GST.
        #
        #    `self.root` is splitted into two dirs (note the impact on
        #    `TileDataset`):
        #    `self.raw_dir`: `self.root`/"raw"
        #    `self.processed_dir`: `self.root`/"processed"
        #
        #    `dowaload` method can be ignored.
        #
        #    Parameters:
        #        n_nodes_per_seg: number of nodes per graph segment
        #    """
        #
        #    def __init__(
        #        self,
        #        root: Optional[str] = "./data/",
        #        src: str = "nlp",  # "nlp" or "xla"
        #        search: str = "random",  # "random" or "default"
        #        n_nodes_per_seg: int = 1000,
        #        transform: Optional[Callable] = None,
        #        pre_transform: Optional[Callable] = None,
        #        pre_filter: Optional[Callable] = None,
        #        log: bool = True
        #    ) -> None:
        #
        #        self.src = src
        #        self.search = search
        #        self.n_nodes_per_seg = n_nodes_per_seg
        #
        #        super().__init__(root, transform, pre_transform, pre_filter, log)
        #        self.data, self.slices = torch.load(self.processed_paths[0])
        #
        #        # ===
        #        # If my own scale trafo (e.g., log) is applied, there's no need to norm here
        #        #op_feats_mean = torch.mean(self.data.op_feats, dim=0, keepdim=True)
        #        #op_feats_std = torch.std(self.data.op_feats, dim=0, keepdim=True)
        #        #op_feats_std[op_feats_std < 1e-6] = 1
        #        #self.data.op_feats = (self.data.op_feats - op_feats_mean) / op_feats_std
        #        # ===
        #
        #    @property
        #    def raw_file_names(self) -> List[str]:
        #        """Return files needed to be found in `self.raw_dir` to skip
        #        download.
        #        """
        #        return [f"npz_all/npz/layout/{self.src}/{self.search}"]
        #
        #    @property
        #    def processed_file_names(self) -> List[str]:
        #        """Return files needed to be found in `self.processed_dir` to
        #        skip processing.
        #        """
        #        return [f"data_segment_{self.n_nodes_per_seg}.pt", f"split_dict_segment_{self.n_nodes_per_seg}.pt"]
        #
        #    def process(self) -> None:
        #        """Process the raw data and save it into the processed dir."""
        #        data_list = []
        #        split_dict = {split: [] for split in SPLIT}
        #        graph_cnt = 0
        #        seg_cnt = 0
        #        for raw_path in self.raw_paths:
        #            # For only the current source-search
        #            for split in SPLIT:
        #                data_root = os.path.join(raw_path, split)
        #                for file in os.listdir(data_root):
        #                    if not file.endswith("npz"): continue
        #
        #                    split_dict[split].append(graph_cnt)
        #                    # Load graph data
        #                    file_path = os.path.join(data_root, file)
        #                    layout_tmp = dict(np.load(file_path))
        #                    if "edge_index" not in layout_tmp:
        #                        logging.info(f"{file_path} has no edge_index.")
        #
        #                    # Parse data fields
        #                    node_feat = torch.tensor(layout_tmp["node_feat"])   # (n, 140)
        #                    node_opcode = torch.tensor(layout_tmp["node_opcode"])  # (n, )
        #                    edge_index = torch.tensor(layout_tmp["edge_index"].T)  # (2, m)
        #                    node_config_feat = torch.tensor(layout_tmp["node_config_feat"])
        #                    node_config_feat = node_config_feat.view(-1, node_config_feat.shape[-1])  # (c * nc, 18)
        #
        #                    node_config_ids = torch.tensor(layout_tmp["node_config_ids"])  # (nc, )
        #                    runtime = torch.tensor(layout_tmp["config_runtime"])  # (c,)
        #
        #                    # Derive simple graph stats
        #                    n_nodes = torch.tensor(node_feat.shape[0])
        #                    n_configs = torch.tensor(layout_tmp["node_config_feat"].shape[0])
        #                    n_config_nodes = torch.tensor(layout_tmp["node_config_feat"].shape[1])
        #
        #                    # Segment the whole graph
        #                    n_segs = n_nodes // self.n_nodes_per_seg + 1
        #                    n_nodes_per_seg = n_nodes // n_segs
        #                    seg_ptr = torch.arange(0, n_nodes, n_nodes_per_seg+1)
        #                    if seg_ptr[-1] != n_nodes:
        #                        # Include all the rest nodes
        #                        seg_ptr = torch.cat([seg_ptr, torch.tensor([n_nodes])])
        #
        #                    # Build graph data object
        #                    data = Data(
        #                        node_feat=node_feat,
        #                        node_opcode=node_opcode,
        #                        edge_index=edge_index,
        #                        node_config_feat=node_config_feat,
        #                        node_config_ids=node_config_ids,
        #                        runtime=runtime,
        #                        n_nodes=n_nodes,
        #                        n_configs=n_configs,
        #                        n_config_nodes=n_config_nodes,
        #                        seg_ptr=seg_ptr,  # Indicate segment break points (n_segs + 1)
        #                        partition_idx=seg_cnt
        #                    )
        #                    data_list.append(data)
        #
        #                    graph_cnt += 1
        #                    # ===
        #                    # Why * n_configs???
        #                    seg_cnt += n_segs * n_configs
        #                    # ===
        #
        #            data, slices = self.collate(data_list)
        #            torch.save((data, slices), self.processed_paths[0])
        #            torch.save(split_dict, self.processed_paths[1])
        #
        #    def get_idx_split(self):
        return torch.load(self.processed_paths[1])
