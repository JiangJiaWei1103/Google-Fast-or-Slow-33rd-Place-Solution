"""
Loss criterion building logic.
Author: JiaWei Jiang

This file contains the basic logic of building loss criterion for
training and evaluation processes.
"""
from typing import Any, Dict, Optional

import torch.nn as nn
from torch.nn.modules.loss import _Loss

from .custom import MTL, ListMLE, MultiElementRankLoss, PairwiseHingeLoss


def build_criterion(**loss_params: Any) -> Optional[_Loss]:
    """Build and return the loss criterion.

    In some scenarios, there's a need to define loss criterion along
    with model architecture to enable the access to intermediate
    representation (e.g., multitask learning). In this case, default
    building of loss criterion is disabled and nothing is returned.

    Parameters:
        loss_params: hyperparameters for building loss function

    Return:
        criterion: loss criterion
    """

    def _parse_mtl(loss_fn: str) -> Dict[str, float]:
        loss_fn_with_wt = loss_fn.split(",")
        loss_name2wt = {}
        for loss_str in loss_fn_with_wt:
            loss_name, wt = loss_str.split(":")
            loss_name2wt[loss_name] = float(wt)

        return loss_name2wt

    # Setup configuration
    loss_fn = loss_params["name"]

    criterion: _Loss
    if loss_fn == "l1":
        criterion = nn.L1Loss()
    elif loss_fn == "l2":
        criterion = nn.MSELoss()
    elif loss_fn == "hinge":
        criterion = PairwiseHingeLoss()
    elif loss_fn == "mulelerank":
        criterion = MultiElementRankLoss()
    elif loss_fn == "listmle":
        criterion = ListMLE()
    elif "," in loss_fn:
        loss_name2wt = _parse_mtl(loss_fn)
        criterion = MTL(loss_name2wt)
    elif loss_fn == "mtk":
        print("Loss criterion default building is disabled...")
        criterion = None
    else:
        raise RuntimeError(f"Loss criterion {loss_fn} isn't registered...")

    return criterion
