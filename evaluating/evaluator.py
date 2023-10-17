"""
Evaluator definition.
Author: JiaWei Jiang

This file contains the definition of evaluator used during evaluation
process.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class Evaluator(object):
    """Custom evaluator.

    Following is a simple illustration of evaluator used in regression
    task.

    Parameters:
        metric_names: evaluation metrics
    """

    eval_metrics: Dict[str, Callable[..., Union[float]]] = {}
    EPS: float = 1e-6

    def __init__(self, metric_names: List[str]) -> None:
        self.metric_names = metric_names

        self._build()

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: List[np.ndarray],
        scaler: Optional[object] = None,
    ) -> Dict[str, float]:
        """Run evaluation using pre-specified metrics.

        Parameters:
            y_true: groundtruths
            y_pred: predicting values
            scaler: scaling object
                *Note: For fair comparisons among experiments using
                    models trained on y with different scales, the
                    inverse tranformation is needed.

        Return:
            eval_result: evaluation performance report
        """
        if scaler is not None:
            # Do inverse transformation to rescale y values
            y_pred, y_true = self._rescale_y(y_pred, y_true, scaler)

        eval_result = {}
        for metric_name, metric in self.eval_metrics.items():
            eval_result[metric_name] = metric(y_pred, y_true)

        return eval_result

    def _build(self) -> None:
        """Build evaluation metric instances."""
        for metric_name in self.metric_names:
            # ==Ranking==
            if metric_name == "one_minus_slowdown":
                self.eval_metrics[metric_name] = self._one_minus_slowdown_at_k
            elif metric_name == "ndcg":
                self.eval_metrics[metric_name] = self._ndcg_at_k  # type: ignore
            elif metric_name == "opa":
                self.eval_metrics[metric_name] = self._opa
            elif metric_name == "kendall_tau":
                self.eval_metrics[metric_name] = self._kendall_tau
            # ==Regression==
            elif metric_name == "rmse":
                self.eval_metrics[metric_name] = self._RMSE
            elif metric_name == "mae":
                self.eval_metrics[metric_name] = self._MAE
            elif metric_name == "rrse":
                self.eval_metrics[metric_name] = self._RRSE
            elif metric_name == "rae":
                self.eval_metrics[metric_name] = self._RAE
            elif metric_name == "corr":
                self.eval_metrics[metric_name] = self._CORR

    def _rescale_y(self, y_pred: Tensor, y_true: Tensor, scaler: Any) -> Tuple[Tensor, Tensor]:
        """Rescale y to the original scale.

        Parameters:
            y_pred: predicting results
            y_true: groundtruths
            scaler: scaling object

        Return:
            y_pred: rescaled predicting results
            y_true: rescaled groundtruths
        """
        assert y_pred.shape == y_true.shape, "Shape of prediction must match that of groundtruth."
        if y_pred.dim() == 3:
            n_samples, n_horiz, n_series = y_pred.shape  # B, Q, N
            y_pred = y_pred.reshape(n_samples * n_horiz, -1)
            y_true = y_true.reshape(n_samples * n_horiz, -1)
        else:
            n_horiz = 1

        # Inverse transform
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)

        if n_horiz != 1:
            y_pred = y_pred.reshape(n_samples, n_horiz, -1)
            y_true = y_true.reshape(n_samples, n_horiz, -1)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.float32)

        return y_pred, y_true

    # Ranking metrics
    def _one_minus_slowdown_at_k(self, y_pred: Tensor, y_true: np.ndarray, k: int = 5) -> float:
        """One minus slowdown incurred of the top-k predictions.

        The larger, the better.

        Parameters:
            y_pred: predicting results
            y_true: groundtruths

        Return:
            one_minus_slowdown_at_k: how closer the best time among the
                top-k predictions is to the actual fastest
        """
        n_samples = len(y_true)
        # assert len(y_pred) == n_samples, "`y_pred` length isn't aligned with `y_true`."

        one_minus_slowdown_at_k = 0.0
        ch, ct = 0, None
        for i, gt in enumerate(y_true):
            # Derive config head and tail
            n_configs = len(gt.reshape((-1, 1)))
            ct = ch + n_configs
            y_pred_i = y_pred[ch:ct]
            ch = ct

            best_runtime = np.min(gt)
            best_runtime_pred = np.min(gt[np.argsort(y_pred_i)[:k]])

            one_minus_slowdown_at_k += 2 - best_runtime_pred / best_runtime
        one_minus_slowdown_at_k /= n_samples

        return one_minus_slowdown_at_k

    def _ndcg_at_k(
        self,
        y_pred: Tensor,
        y_true: np.ndarray,
        ks: List[int] = [5, 10, 20, 50],
    ) -> Union[Dict[int, float], float]:
        """Normalized discounted cumulative gain at k.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            ndcg_at_k
        """

        def _dcg_at_k(rel: np.ndarray, y_pred_i: np.ndarray, k_adj: int) -> float:
            topk_idx = np.argsort(y_pred_i)[:k_adj]
            dcg = rel[topk_idx] / np.log2(np.arange(1, k_adj + 1) + 1)
            dcg = np.sum(dcg)

            return dcg

        def _idcg_at_k(rel: np.ndarray, gt: np.ndarray, k_adj: int) -> float:
            topk_idx = np.argsort(gt)[:k_adj]
            idcg = rel[topk_idx] / np.log2(np.arange(1, k_adj + 1) + 1)
            idcg = np.sum(idcg)

            return idcg

        # ===
        k = 10
        # ===
        n_samples = len(y_true)
        # assert len(y_pred) == n_samples, "`y_pred` length isn't aligned with `y_true`."

        ndcg_at_k = 0.0
        ch, ct = 0, None
        for gt in y_true:
            # Derive config head and tail
            n_configs = len(gt.reshape((-1, 1)))
            ct = ch + n_configs
            y_pred_i = y_pred[ch:ct]
            ch = ct

            rel = 1 / gt
            k_adj = min(k, n_configs)
            ndcg = _dcg_at_k(rel, y_pred_i, k_adj) / _idcg_at_k(rel, gt, k_adj)
            ndcg_at_k += ndcg
        ndcg_at_k /= n_samples

        return ndcg_at_k

    def _opa(self, y_pred: Tensor, y_true: np.ndarray) -> float:
        """Ordered pair accuracy.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            opa: ordered pair accuracy
        """
        n_samples, n_configs = y_true.shape  # (b, c)
        n_concords, n_discords = self._cal_condord_discord(y_pred, y_true)
        opa = torch.mean(n_concords / (n_concords + n_discords + self.EPS)).item()
        # i_idx = torch.arange(n_configs).repeat(n_configs)
        # j_idx = torch.arange(n_configs).repeat_interleave(n_configs)
        # pairwise_y_true = y_true[:, i_idx] > y_true[:, j_idx]
        # pairwise_y_pred = y_pred[:, i_idx] > y_pred[:, j_idx]
        # numer = torch.sum(pairwise_y_pred & pairwise_y_true, dim=1)
        # denom = torch.sum(pairwise_y_true, dim=1)
        # assert len(numer) == n_samples and len(numer) == len(denom), "#Samples isn't aligned for _opa."
        # opa = torch.mean(numer / denom)

        return opa

    def _kendall_tau(self, y_pred: Tensor, y_true: np.ndarray) -> float:
        """Kendall tau correlation.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            kendall_tau: kendall tau correlation
        """
        n_samples, n_configs = y_true.shape  # (b, c)
        n_concords, n_discords = self._cal_condord_discord(y_pred, y_true)
        kendall_tau = torch.mean((n_concords - n_discords) / (n_concords + n_discords + self.EPS)).item()

        return kendall_tau

    def _cal_condord_discord(self, y_pred: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate number of concordant and discordant pairs.

        For pairwise comparison in the matrix form, we know that -A == A.T,
        so we only need to consider one of the triangle.

        Return:
            n_concords: number of concardant pairs
            n_discords: number of discordant pairs

        Shape:
            y_pred: (B, C), where C denotes the number of configurations
            y_true: (B, C)
            n_concords: (B, )
            n_discords: (B, )
        """
        # Item is configuration in this case
        n_items = y_true.shape[1]

        tril_mask = torch.ones((n_items, n_items), device=y_true.device).tril(diagonal=-1)
        pairwise_y_true_diff = y_true.unsqueeze(dim=-1) - y_true.unsqueeze(dim=1)
        pairwise_y_pred_diff = y_pred.unsqueeze(dim=-1) - y_pred.unsqueeze(dim=1)
        n_concords = (((pairwise_y_true_diff * pairwise_y_pred_diff) > 0) * tril_mask).sum(dim=[1, 2])
        n_discords = (((pairwise_y_true_diff * pairwise_y_pred_diff) < 0) * tril_mask).sum(dim=[1, 2])

        return n_concords, n_discords

    # Regression metrics
    def _RMSE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Root mean squared error.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            rmse: root mean squared error
        """
        mse = nn.MSELoss()
        rmse = torch.sqrt(mse(y_pred, y_true)).item()

        return rmse

    def _MAE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Mean absolute error.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            mae: root mean squared error
        """
        mae = nn.L1Loss()(y_pred, y_true).item()

        return mae

    def _RRSE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Root relative squared error.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            rrse: root relative squared error
        """
        #         gt_mean = torch.mean(y_true)
        #         sse = nn.MSELoss(reduction="sum")  # Sum squared error
        #         rrse = torch.sqrt(
        #             sse(y_pred, y_true) / sse(gt_mean.expand(y_true.shape), y_true)
        #         ).item()
        mse = nn.MSELoss()
        rrse = (torch.sqrt(mse(y_pred, y_true)) / torch.std(y_true)).item()

        return rrse

    def _RAE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Relative absolute error.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            rae: relative absolute error
        """
        gt_mean = torch.mean(y_true)

        sae = nn.L1Loss(reduction="sum")  # Sum absolute error
        rae = (sae(y_pred, y_true) / sae(gt_mean.expand(y_true.shape), y_true)).item()

        return rae

    def _CORR(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Empirical correlation coefficient.

        Because there are some time series with zero values across the
        specified dataset (e.g., time series idx 182 in electricity
        across val and test set with size of splitting 6:2:2), corr of
        such series are dropped to avoid situations like +-inf or NaN.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            corr: empirical correlation coefficient
        """
        pred_mean = torch.mean(y_pred, dim=0)
        pred_std = torch.std(y_pred, dim=0)
        gt_mean = torch.mean(y_true, dim=0)
        gt_std = torch.std(y_true, dim=0)

        # Extract legitimate time series index with non-zero std to
        # avoid situations stated in *Note.
        gt_idx_leg = gt_std != 0
        idx_leg = gt_idx_leg
        #         pred_idx_leg = pred_std != 0
        #         idx_leg = torch.logical_and(pred_idx_leg, gt_idx_leg)

        corr_per_ts = torch.mean(((y_pred - pred_mean) * (y_true - gt_mean)), dim=0) / (pred_std * gt_std)
        corr = torch.mean(corr_per_ts[idx_leg]).item()  # Take mean across time series

        return corr
