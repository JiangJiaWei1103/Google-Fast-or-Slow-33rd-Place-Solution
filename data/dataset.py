"""
Dataset definitions.
Author: JiaWei Jiang

This file contains definitions of multiple datasets used in different
scenarios.

* [ ] Add raw runtime as aux task (e.g., log runtime w/o norm).
"""
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

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
            data_sample["target"] = torch.tensor(data_row["target_norm"], dtype=torch.float32)

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
