"""
Custom trainer definitions for different training processes.
Author: JiaWei Jiang

This file contains diversified trainers, whose training logics are
inherited from `BaseTrainer`.

* [ ] Pack input data in Dict.
* [ ] Fuse grad clipping mechanism into solver.
* [ ] Clarify grad accumulation w/ clipping.
* [ ] Grad accumulation with drop_last or updating with the remaining
    samples.
"""
import copy
import gc
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_sparse import SparseTensor
from tqdm import tqdm

from base.base_trainer import BaseTrainer
from evaluating.evaluator import Evaluator
from metadata import NODE_CONFIG_FEAT_DIM
from modeling.baseline.LayoutBaseline import HistoryEmbTable
from utils.early_stopping import EarlyStopping


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

        self.loss_name = self.loss_fn.__class__.__name__

        self.grad_accum_steps = proc_cfg["solver"]["optimizer"]["grad_accum_steps"]

    def _train_epoch(self) -> float:
        """Run training process for one epoch.

        Return:
            train_loss_avg: average training loss over batches
        """
        train_loss_total = 0

        self.model.train()
        self.profiler.start("train")
        for i, batch_data in enumerate(tqdm(self.train_loader)):
            if i % self.grad_accum_steps == 0:
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

            # Derive loss
            if self.loss_name in ["ListMLE", "MultiElementRankLoss"]:
                loss = self.loss_fn(output, y, inputs["n_configs"])
            else:
                loss = self.loss_fn(output, y)
            train_loss_total += loss.item()
            loss = loss / self.grad_accum_steps

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-3)

            if (i + 1) % self.grad_accum_steps == 0:
                self.optimizer.step()
                if self.step_per_batch:
                    self.lr_skd.step()

            self._iter += 1
            # train_loss_total += loss.item()

            # ===
            # if self.step_per_batch:
            #    self.lr_skd.step()
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
            if self.loss_name in ["ListMLE", "MultiElementRankLoss"]:
                loss = self.loss_fn(output, y, inputs["n_configs"])
            else:
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


class GSTrainer(BaseTrainer):
    """Trainer used for Graph Segment Training (GST).

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
        super(GSTrainer, self).__init__(
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

        self.loss_name = self.loss_fn.__class__.__name__

        self.grad_accum_steps = proc_cfg["solver"]["optimizer"]["grad_accum_steps"]

        # Configuration sampler
        self.config_sampler = _ConfigSampler(n_configs=32, include_extremes=True)

        # Historical embedding table
        self.hetable = HistoryEmbTable(int(6e7), 1, self.device)

    def _train_epoch(self) -> float:
        """Run training process for one epoch.

        Return:
            train_loss_avg: average training loss over batches
        """
        train_loss_total = 0

        self.model.train()
        self.profiler.start("train")
        for i, batch_data in enumerate(tqdm(self.train_loader)):
            if i % self.grad_accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            batch_data, batch_sampled_config_idx = self.config_sampler.sample(batch_data)
            y = batch_data.y.to(self.device)
            batch_to_train, batch_n_segs, batch_other, segs_to_train = self._rebatch_by_seg(
                batch_data, batch_sampled_config_idx
            )

            # Forward pass
            output, x_graph = self.model(batch_to_train, batch_other, batch_n_segs)

            # Derive loss
            output = output.view(-1, self.config_sampler.n_configs)  # (n_graphs, 32)
            y = y.view(-1, self.config_sampler.n_configs)
            loss = self.loss_fn(output, y)
            train_loss_total += loss.item()
            loss = loss / self.grad_accum_steps

            # Backpropagation
            # Clip where ???????????????
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-3)

            if (i + 1) % self.grad_accum_steps == 0 or i + 1 == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-2)  # 1.0)
                self.optimizer.step()
                if self.step_per_batch:
                    self.lr_skd.step()

            # Push update-to-date segment embedding to hash table
            batch_data_list = batch_data.to_data_list()
            n_sampled_configs = self.config_sampler.n_configs
            seg_embs = x_graph.reshape(-1, n_sampled_configs)
            for b, seg_emb in enumerate(seg_embs):
                parent_graph = batch_data_list[b]
                hash_head, n_segs = parent_graph.hash_head.item(), parent_graph.n_segs.item()
                seg_to_train = segs_to_train[b]
                for k, seg_emb_entry in enumerate(seg_emb):
                    hash_idx = hash_head + seg_to_train + batch_sampled_config_idx[b][k] * n_segs
                    self.hetable.push(seg_emb_entry, hash_idx)

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
        y_true, y_pred = None, None

        self.model.eval()
        self.profiler.start(proc_type=datatype)
        for i, batch_data in enumerate(self.eval_loader):
            batch_data, _ = self.config_sampler.sample(batch_data)
            y = batch_data.y.to(self.device)
            batch_to_eval, batch_n_segs, *_ = self._rebatch_by_seg(batch_data, None, train=False)

            # Forward pass
            _, x_graph = self.model(batch_to_eval)  # x_graph as output

            # Derive loss
            y = y.view(-1, self.config_sampler.n_configs).to(self.device)
            n_sampled_configs = self.config_sampler.n_configs
            output_seg = x_graph.reshape(-1, n_sampled_configs)
            batch_n_segs_idx = torch.repeat_interleave(torch.arange(len(batch_n_segs)), torch.tensor(batch_n_segs)).to(
                y.device
            )
            batch_n_segs_idx = batch_n_segs_idx.reshape(-1, 1).expand(-1, n_sampled_configs)
            output = torch.zeros_like(y, device=y.device, dtype=torch.float32).scatter_add_(
                0, batch_n_segs_idx, output_seg
            )
            loss = self.loss_fn(output, y)
            eval_loss_total += loss.item()

            if i == 0:
                y_true = y.detach().cpu()
                y_pred = output.detach().cpu()
            else:
                y_true = torch.cat([y_true, y.detach().cpu()], dim=0)
                y_pred = torch.cat([y_pred, output.detach().cpu()], dim=0)

        self.profiler.stop(record=True if datatype != "train" else False)
        eval_loss_avg = eval_loss_total / len(self.eval_loader)
        eval_result = self.evaluator.evaluate(y_true, y_pred, self.scaler)

        if return_output:
            return eval_loss_avg, eval_result, y_pred
        else:
            return eval_loss_avg, eval_result, None

    def _rebatch_by_seg(
        self, batch_data: Batch, batch_sampled_config_idx: Optional[List[Tensor]] = None, train: bool = True
    ) -> Tuple[Batch, List[int], Optional[List[Tensor]], Optional[List[int]]]:
        """Rebatch data based on graph segments."""
        if train:
            assert (
                batch_sampled_config_idx is not None
            ), "For training, please provide batch_sampled_config_idx for hashing."
        batch_data_list = batch_data.to_data_list()
        batch_data = []
        batch_n_segs = []
        batch_other = []
        segs_to_train = []
        for i, g in enumerate(batch_data_list):
            seg_ptr, n_segs = g.seg_ptr, g.n_segs.item()
            if train:
                batch_n_segs.extend([n_segs] * self.config_sampler.n_configs)
                seg_to_train = np.random.randint(n_segs)  # Only one seg is selected, S^(i) = 1
                segs_to_train.append(seg_to_train)
            else:
                batch_n_segs.append(n_segs)
                seg_to_train = None
            for j in range(n_segs):
                seg_h, seg_t = seg_ptr[j].item(), seg_ptr[j + 1].item()
                seg_size = seg_t - seg_h

                n, m = g.n_nodes.item(), g.n_edges.item()
                g_seg = copy.copy(g)
                for k, v in g:
                    if isinstance(v, Tensor) and v.size(0) == n:
                        # Narrow node-level attr
                        g_seg[k] = v.narrow(0, seg_h, seg_size)
                    else:
                        g_seg[k] = v
                adj = g_seg.adj.narrow(0, seg_h, seg_size).narrow(1, seg_h, seg_size)
                row, col, _ = adj.coo()
                g_seg.edge_index = torch.stack([row, col], dim=0)

                if train and j != seg_to_train:
                    for k in range(self.config_sampler.n_configs):
                        hash_idx = g.hash_head.item() + n_segs * batch_sampled_config_idx[i][k].item() + j
                        batch_other.append(self.hetable.pull(hash_idx))
                else:
                    for k in range(self.config_sampler.n_configs):
                        # Construct data sample for each seg-config pair
                        g_seg_config = Data(
                            edge_index=g_seg.edge_index,
                            node_feat=g_seg.node_feat,
                            node_opcode=g_seg.node_opcode,
                            # g_seg.node_config_feat: (n, 32, CONFIG_FEAT_DIM)
                            node_config_feat=g_seg.node_config_feat[:, k, :],
                            n_nodes=seg_size,
                        )
                        batch_data.append(g_seg_config)

        batch_data = Batch.from_data_list(batch_data).to(self.device)
        if not train:
            batch_other, segs_to_train = None, None

        return batch_data, batch_n_segs, batch_other, segs_to_train


class _ConfigSampler(object):
    """Configuration sampler.

    Support random sampling now.
    """

    def __init__(self, n_configs: int = 32, include_extremes: bool = True) -> None:
        self.n_configs = n_configs
        self.include_extremes = include_extremes

    def sample(self, batch: Batch) -> Tuple[Batch, List[Tensor]]:
        batch_data_list = batch.to_data_list()
        processed_batch_list = []
        batch_sampled_config_idx = []
        for i, g in enumerate(batch_data_list):
            # print(f"Sample graph{i}... with #configs = {g.n_configs}, #segs = {g.n_segs}")
            n_nodes, n_config_nodes, n_configs = g.n_nodes.item(), g.n_config_nodes.item(), g.n_configs.item()
            sampled_config_idx = torch.randint(0, n_configs, (self.n_configs,))

            # Narrow attributes along config dimension
            g.y = g.y[sampled_config_idx]
            g.node_config_feat = g.node_config_feat.view(n_configs, n_config_nodes, -1)[
                sampled_config_idx, ...
            ].transpose(
                0, 1
            )  # (nc, self.n_configs, CONFIG_FEAT_DIM)
            g.node_config_feat_full = torch.zeros(n_nodes, self.n_configs, NODE_CONFIG_FEAT_DIM, dtype=torch.float32)
            g.node_config_feat_full[g.node_config_ids, ...] += g.node_config_feat
            g.node_config_feat, g.node_config_feat_full = g.node_config_feat_full, None
            g.adj = SparseTensor(row=g.edge_index[0], col=g.edge_index[1], sparse_sizes=(n_nodes, n_nodes))
            processed_batch_list.append(g)
            batch_sampled_config_idx.append(sampled_config_idx.view(1, -1))
        processed_batch = Batch.from_data_list(processed_batch_list)
        batch_sampled_config_idx = torch.cat(batch_sampled_config_idx, dim=0)

        return processed_batch, batch_sampled_config_idx
