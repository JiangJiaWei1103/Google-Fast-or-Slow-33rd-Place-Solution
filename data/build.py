"""
Dataloader building logic.
Author: JiaWei Jiang

This file contains the basic logic of building dataloaders for training
and evaluation processes.

* [ ] Modify dataloader building logic (maybe one at a time).
"""
from typing import Any, Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader

from .collate import tile_collate
from .dataset import TileDataset


def build_dataloaders(
    data_tr: pd.DataFrame,
    data_eval: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    **dataset_cfg: Any,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create and return train and validation data loaders.

    Parameters:
        data_tr: training data
        data_eval: evaluation data, either validation or test
        batch_size: number of samples per batch
        shuffle: whether to shuffle samples every epoch
        num_workers: number of subprocesses used to load data
        dataset_cfg: hyperparameters of customized dataset

    Return:
        train_loader: training data loader
        eval_loader: evaluation data loader
    """
    # collate_fn = None
    collate_fn = tile_collate

    train_loader = DataLoader(
        TileDataset(data_tr, **dataset_cfg),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    if data_eval is not None:
        eval_loader = DataLoader(
            TileDataset(data_eval, **dataset_cfg),
            batch_size=batch_size,  # * k
            shuffle=False,  # Hard-coded
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        return train_loader, eval_loader
    else:
        return train_loader, None
