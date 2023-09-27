"""
Custom trainer definitions for different training processes.
Author: JiaWei Jiang

This file contains diversified trainers, whose training logics are
inherited from `BaseTrainer`.

* [ ] Pack input data in Dict.
* [ ] Fuse grad clipping mechanism into solver.
"""
import gc
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.base_trainer import BaseTrainer
from evaluating.evaluator import Evaluator
from utils.early_stopping import EarlyStopping

# from utils.scaler import MaxScaler, StandardScaler


class MainTrainer(BaseTrainer):
    """Main trainer.

    It's better to define different trainers for different models if
    there's a significant difference within training and evaluation
    processes (e.g., model input, advanced data processing, graph node
    sampling, customized multitask criterion definition).

    Parameters:
        logger: message logger
        proc_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_scheduler: learning rate scheduler
        es: early stopping tracker
        scaler: scaling object
        train_loader: training data loader
        eval_loader: validation data loader
        use_wandb: if True, training and evaluation processes are
            tracked with WandB
    """

    def __init__(
        self,
        logger: Logger,
        proc_cfg: Dict[str, Any],
        model: Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau],
        ckpt_path: Union[Path, str],
        es: EarlyStopping,
        evaluator: Evaluator,
        scaler: Any,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        use_wandb: bool = True,
    ):
        super(MainTrainer, self).__init__(
            logger,
            proc_cfg,
            model,
            loss_fn,
            optimizer,
            lr_skd,
            ckpt_path,
            es,
            evaluator,
            use_wandb,
        )
        self.train_loader = train_loader
        self.eval_loader = eval_loader if eval_loader else train_loader
        self.scaler = scaler
        self.rescale = proc_cfg["loss_fn"]["rescale"]

    def _train_epoch(self) -> float:
        """Run training process for one epoch.

        Return:
            train_loss_avg: average training loss over batches
        """
        train_loss_total = 0

        self.model.train()
        self.profiler.start("train")
        for i, batch_data in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad(set_to_none=True)

            # Retrieve batched raw data
            inputs = {}
            for k, v in batch_data.items():
                if k != "target":
                    inputs[k] = v.to(self.device)
                else:
                    y = v.to(self.device)

            # Forward pass and derive loss
            output = self.model(inputs)

            # Inverse transform to the original scale
            # if self.rescale:
            #    output = self.scaler.inverse_transform(output)
            #    y = self.scaler.inverse_transform(y)

            # Derive loss
            loss = self.loss_fn(output, y)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-3)

            self.optimizer.step()
            self._iter += 1
            train_loss_total += loss.item()

            # ===
            self.lr_skd.step()
            # ===

            # Free mem.
            del inputs, y, output
            _ = gc.collect()

        self.profiler.stop()
        train_loss_avg = train_loss_total / len(self.train_loader)

        return train_loss_avg

    @torch.no_grad()
    def _eval_epoch(
        self,
        return_output: bool = False,
        datatype: str = "val",
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Parameters:
            return_output: whether to return inference result of model
            datatype: type of the dataset to evaluate

        Return:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: inference result
        """
        eval_loss_total = 0
        y_pred = []

        self.model.eval()
        self.profiler.start(proc_type=datatype)
        for i, batch_data in enumerate(self.eval_loader):
            # Retrieve batched raw data
            inputs = {}
            for k, v in batch_data.items():
                if k != "target":
                    inputs[k] = v.to(self.device)
                else:
                    y = v.to(self.device)

            # Forward pass
            output = self.model(inputs)

            # Derive loss
            loss = self.loss_fn(output, y)
            eval_loss_total += loss.item()

            # Record batched output
            y_pred.append(output.detach().cpu())

            del inputs, y, output
            _ = gc.collect()

        self.profiler.stop(record=True if datatype != "train" else False)
        eval_loss_avg = eval_loss_total / len(self.eval_loader)

        # Run evaluation with the specified evaluation metrics
        y_true = self.eval_loader.dataset.data["target"].values
        y_pred = torch.cat(y_pred, dim=-1)
        eval_result = self.evaluator.evaluate(y_true, y_pred, self.scaler)

        if return_output:
            return eval_loss_avg, eval_result, y_pred
        else:
            return eval_loss_avg, eval_result, None
