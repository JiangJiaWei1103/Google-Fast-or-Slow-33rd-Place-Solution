"""
Custom criterions.
Author: JiaWei Jiang
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.nn.utils.rnn import pad_sequence


class BaseRankLoss(_Loss):

    pad_token: float = -1e6

    def _split_and_pad(self, y_pred: Tensor, y_true: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Split targets by batch indicator and pad to the max len.

        Parameters:
            y_pred: predicting results
            y_true: groundtruths
            batch: number of configurations of each graph in batch
        """
        max_len = 0
        ch, ct = 0, None
        y_pred_, y_true_ = [], []
        for n_configs in batch:
            ct = ch + n_configs
            y_pred_.append(y_pred[ch:ct])
            y_true_.append(y_true[ch:ct])

            if n_configs > max_len:
                max_len = n_configs
            ch += n_configs

        # Pad to max length
        y_pred_ = pad_sequence(y_pred_, True, self.pad_token)
        y_true_ = pad_sequence(y_true_, True, self.pad_token)

        return y_pred_, y_true_


class MultiElementRankLoss(BaseRankLoss):
    def __init__(self, margin: float = 0.1, n_permuts: int = 4) -> None:
        super().__init__()

        self.loss_fn = torch.nn.MarginRankingLoss(margin=margin, reduction="none")
        self.n_permuts = n_permuts

    def _cal_rank_loss(
        self,
        y_pred: Tensor,
        y_true: Tensor,
    ) -> Tensor:
        """
        Generates a permutation of the predictions and targets and
        calculates the loss MarginRankingLoss against the permutation.

        Parameters:
            y_pred: Tensor of shape (bs, seq_len) with the y_pred of the model
            y_true: Tensor of shape (bs, seq_len) with the runtime of the model
            config_mask: Tensor of shape (bs, seq_len) with 1 in the positions of the elements
            and 0 in the positions of the padding

        Return:
            loss: Tensor of shape (bs, seq_len) with the loss for each element in the batch
        """
        bs, num_configs = y_pred.shape

        config_idxs = torch.arange(num_configs).unsqueeze(0).expand(bs, -1).to(y_pred.device)
        # permuted_idxs = torch.argsort(torch.rand_like(config_idxs), dim=-1)

        permutation = torch.randperm(num_configs)
        permuted_idxs = config_idxs[:, permutation]

        # We mask those cases where we compare the same configuration
        config_mask = torch.where(config_idxs != permuted_idxs, 1, 0)
        permuted_runtime = y_true[:, permutation]
        labels = 2 * ((y_true - permuted_runtime) > 0) - 1
        permuted_output = y_pred[:, permutation]
        loss = self.loss_fn(y_pred.view(-1, 1), permuted_output.view(-1, 1), labels.view(-1, 1))
        loss = loss.view(bs, num_configs) * config_mask * (permuted_runtime != self.pad_token)

        return loss.mean()

    def forward(self, y_pred: Tensor, y_true: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        # tile needs batch, layout no
        # y_pred, y_true = self._split_and_pad(y_pred, y_true, batch)

        loss = 0
        for _ in range(self.n_permuts):
            loss += self._cal_rank_loss(y_pred, y_true)
        loss = loss / self.n_permuts

        return loss


class MTL(_Loss):
    """Multitask loss."""

    def __init__(self, loss_name2wt: Dict[str, float]) -> None:
        super().__init__()

        self.loss_name2wt = loss_name2wt
        self.loss_fn = {}
        for loss_name, wt in loss_name2wt.items():
            if loss_name == "hinge":
                self.loss_fn[loss_name] = (PairwiseHingeLoss(), wt)
            elif loss_name == "l1":
                self.loss_fn[loss_name] = (nn.L1Loss(), wt)
            elif loss_name == "l2":
                self.loss_fn[loss_name] = (nn.MSELoss(), wt)

    def forward(self, y_pred: Tensor, y_true: Tensor, y_pred_reg: Tensor) -> Dict[str, Tensor]:
        loss = {}
        tot_loss = 0
        for loss_name, (loss_fn, wt) in self.loss_fn.items():
            if loss_name == "hinge":
                loss[loss_name] = loss_fn(y_pred, y_true)
            elif y_pred_reg is not None:
                loss[loss_name] = loss_fn(y_pred_reg, y_true)
            else:
                loss[loss_name] = 0
            tot_loss = tot_loss + loss[loss_name] * wt
        loss["loss"] = tot_loss

        return loss


class PairwiseHingeLoss(_Loss):
    """Pairwise hinge loss."""

    def __init__(self) -> None:
        super().__init__()

        self.eps = 0.1  # GST
        # self.eps = 1  # tf

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        batch_size, num_preds = y_pred.shape
        i_idx = torch.arange(num_preds).repeat(num_preds)
        j_idx = torch.arange(num_preds).repeat_interleave(num_preds)
        pairwise_true = y_true[:, i_idx] > y_true[:, j_idx]
        loss = (
            torch.sum(
                torch.nn.functional.relu(self.eps - (y_pred[:, i_idx] - y_pred[:, j_idx])) * pairwise_true.float()
            )
            / batch_size
        )

        return loss


class ListMLE(_Loss):

    EPS: float = 1e-10

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

        observation_loss = torch.log(cumsums + self.EPS) - preds_sorted_by_true_minus_max
        loss = torch.mean(torch.sum(observation_loss, dim=1))

        return loss


# class ListMLE(_Loss):
#    """ListMLE loss.
#
#    See https://github.com/allegro/allRank.
#
#    Parameters:
#        temperature: temperature to scale the ranking scores
#        pad_token: padding token
#    """
#
#    EPS: float = 1e-10
#
#    def __init__(
#        self,
#        temperature: float = 10.0,
#        pad_token: float = -1e16
#    ) -> None:
#        super().__init__()
#
#        self.temperature = temperature
#        self.pad_token = pad_token
#
#    def forward(
#        self,
#        y_pred: Tensor,
#        y_true: Tensor,
#        batch: Tensor
#    ) -> Tensor:
#        y_pred = self._get_logits(y_pred)
#        y_pred, y_true = self._split_and_pad(y_pred, y_true, batch)
#
#        # shuffle for randomised tie resolution
#        random_indices = torch.randperm(y_pred.shape[-1])
#        y_pred_shuffled = y_pred[:, random_indices]
#        y_true_shuffled = y_true[:, random_indices]
#
#        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
#
#        mask = y_true_sorted == self.pad_token #padded_value_indicator
#
#        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
#        preds_sorted_by_true[mask] = float("-inf")
#
#        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
#
#        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
#
#        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
#
#        observation_loss = torch.log(cumsums + self.EPS) - preds_sorted_by_true_minus_max
#
#        observation_loss[mask] = 0.0
#
#        return torch.mean(torch.sum(observation_loss, dim=1))
#
#        # Mask out padded entries
#        #mask = y_true == self.pad_token
#        #y_true[mask] = 0.0
#        #y_pred[mask] = math.log(self.EPS)
#        #scores = torch.where(mask, y_true.min(dim=1, keepdim=True)[0] - 1e-6, y_true)
#        #
#        ## Shuffle for breaking tie
#        #rand_idx = torch.randperm(y_pred.shape[-1])
#        #scores_shuffled = scores[:, rand_idx]
#
#        ## Sort scores
#        #_, sort_idx = scores_shuffled.sort(descending=True, dim=-1)
#        #y_pred_sorted_by_true = torch.gather(y_pred, dim=1, index=sort_idx)
#
#        #y_pred_max, _ = y_pred_sorted_by_true.max(dim=1, keepdim=True)
#        #y_pred_sorted_by_true_minus_max = y_pred_sorted_by_true - y_pred_max
#        #cumsums = torch.cumsum(y_pred_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
#
#        #loss = torch.log(cumsums + self.EPS) - y_pred_sorted_by_true_minus_max
#        #loss[mask] = 0.0
#        #loss = torch.mean(torch.sum(loss, dim=1))
#
#    def _get_logits(self, y_pred: Tensor) -> Tensor:
#        logits = y_pred / self.temperature
#
#        return logits
#
