"""
Main script for running evaluation process on layout collections.
Author: JiaWei Jiang
"""
import copy
import gc
import json
import logging
import os
import warnings
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm

from base.base_argparser import BaseArgParser
from data.dataset import LayoutDataset
from evaluating.evaluator import Evaluator
from experiment.experiment import Experiment
from metadata import NODE_CONFIG_FEAT_DIM
from modeling.build import build_model
from paths import PROC_DATA_PATH

warnings.simplefilter("ignore")
LAYOUT_PROC_PATH = Path(PROC_DATA_PATH) / "layout"
DEVICE = torch.device("cuda:0")


class InferArgParser(BaseArgParser):
    """Inference argument parser."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        self.argparser.add_argument("--exp-id", type=str, default=None)
        self.argparser.add_argument(
            "--coll", type=str, choices=["nlp-random", "nlp-default", "xla-random", "xla-default"], default=None
        )
        self.argparser.add_argument("--data-split", type=str, default="valid", choices=["train", "valid", "test"])
        self.argparser.add_argument("--model-name", type=str, default=None)
        self.argparser.add_argument("--mid", type=str, default=None)
        self.argparser.add_argument("--seeds", type=str, nargs="+", default=None)
        self.argparser.add_argument("--use-wandb", type=self._str2bool, help="exists for compatibility", default=False)


@torch.no_grad()
def _run_infer(models: List[Module], dataloader: DataLoader) -> Tuple[List[Tensor], ...]:
    """Run inference using well-trained models.

    Inference is for pure prediction without evaluation.

    Parameters:
        models: well-trained models
        dataloader: data loader

    Return:
        y_pred: prediction
    """
    y_true = []
    y_pred = []
    n_models = len(models)

    logging.info(f"\tStart batchified inference with {n_models} models...")
    models = [model.eval() for model in models]
    for i, batch_data in enumerate(dataloader):  # , total=len(dataloader)):
        logging.info(f"Graph {i}...")

        # One graph at a time
        g = batch_data.to_data_list()[0]
        seg_ptr, n_segs, n_configs = g.seg_ptr, g.n_segs.item(), g.n_configs.item()
        # seg_id, n_segs, n_configs = g.seg_id, g.n_segs.item(), g.n_configs.item()
        n, m = g.n_nodes.item(), g.n_edges.item()
        g.node_config_feat = g.node_config_feat.view(n_configs, -1, NODE_CONFIG_FEAT_DIM)  # (c, nc, 18)
        g.adj = SparseTensor(row=g.edge_index[0], col=g.edge_index[1], sparse_sizes=(n, n))
        y_true.append(g.y)

        x_graph = []
        batch_data_infer = []
        sample_cnt = 0
        for k in tqdm(range(n_configs), total=n_configs):
            node_config_feat_k = g.node_config_feat[k, ...]  # (nc, 18)
            node_config_feat_full = torch.zeros(n, NODE_CONFIG_FEAT_DIM, dtype=torch.float32)
            node_config_feat_full[g.node_config_ids] += node_config_feat_k
            g.node_config_feat_k = node_config_feat_full  # (n, 18)
            del node_config_feat_k, node_config_feat_full
            gc.collect()

            for j in range(n_segs):
                seg_h, seg_t = seg_ptr[j].item(), seg_ptr[j + 1].item()
                seg_size = seg_t - seg_h
                # seg_idx = torch.where(seg_id == j)[0]
                # seg_size = len(seg_id)

                g_seg = copy.copy(g)
                for field, value in g:
                    if isinstance(value, Tensor) and value.size(0) == n:
                        # Narrow node-level attr
                        g_seg[field] = value.narrow(0, seg_h, seg_size)
                        # g_seg[field] = value[seg_idx]
                    else:
                        g_seg[field] = value
                adj = g_seg.adj.narrow(0, seg_h, seg_size).narrow(1, seg_h, seg_size)
                # adj = g_seg.adj[seg_idx, seg_idx]
                row, col, _ = adj.coo()
                g_seg.edge_index = torch.stack([row, col], dim=0)

                g_seg_config = Data(
                    edge_index=g_seg.edge_index,
                    node_feat=g_seg.node_feat,
                    node_opcode=g_seg.node_opcode,
                    node_config_feat=g_seg.node_config_feat_k,
                    n_nodes=seg_size,
                )
                batch_data_infer.append(g_seg_config)
                sample_cnt += 1
                if sample_cnt % 1024 == 0:
                    batch_data_infer = Batch.from_data_list(batch_data_infer).to(DEVICE)
                    for mi, model in enumerate(models):
                        _, x_graph_mi = model(batch_data_infer)  # (256, 1)
                        if mi == 0:
                            x_graph_ = x_graph_mi / n_models
                        else:
                            x_graph_ += x_graph_mi / n_models

                    x_graph.append(x_graph_.detach().cpu())
                    del batch_data_infer, x_graph_, x_graph_mi
                    gc.collect()
                    batch_data_infer = []
        if len(batch_data_infer) > 0:
            batch_data_infer = Batch.from_data_list(batch_data_infer).to(DEVICE)
            for mi, model in enumerate(models):
                _, x_graph_mi = model(batch_data_infer)  # (256, 1)
                if mi == 0:
                    x_graph_ = x_graph_mi / n_models
                else:
                    x_graph_ += x_graph_mi / n_models
            x_graph.append(x_graph_.detach().cpu())
            del batch_data_infer
            gc.collect()

        # Add up outputs from segments
        output_seg = torch.cat(x_graph, dim=0).reshape(n_configs, -1)  # (n_configs, n_segs)
        output = torch.sum(output_seg, dim=1)  # (n_configs, )
        y_pred.append(output)

    return y_true, y_pred


def _gen_sub(data: pd.DataFrame, top_configs: List[str], src: str, search: str) -> pd.DataFrame:
    """Generate submission."""
    sub = pd.read_csv("data/raw/sample_submission.csv")
    data["pred"] = top_configs
    data["ID"] = data["file"].apply(lambda x: f"layout:{src}:{search}:" + x.split(".")[0])
    sub = sub.merge(data[["ID", "pred"]], on="ID", how="left")
    coll_mask = ~sub["pred"].isna()
    sub.loc[coll_mask, "TopConfigs"] = sub.loc[coll_mask, "pred"]
    sub.drop("pred", axis=1, inplace=True)

    return sub


def main(args: Namespace) -> None:
    """Run inference process.

    Parameters:
        args: arguments driving inference process

    Returns:
        None
    """
    # Configure experiment
    experiment = Experiment(args, log_file="infer.log", infer=True)
    coll = args.coll
    src, search = coll.split("-")
    data_split = args.data_split
    mid = args.mid
    seeds = args.seeds

    with experiment as exp:
        # Prepare data
        if data_split == "valid":
            data = pd.read_pickle(LAYOUT_PROC_PATH / f"{src}/{search}/train_new_206.pkl")
            data = data[data["strat_f5_s42_y_mean_log"] == 3].reset_index(drop=True)
        else:
            data = pd.read_pickle(LAYOUT_PROC_PATH / f"{src}/{search}/{data_split}_206.pkl")

        dataloader = DataLoader(
            LayoutDataset(data, **{**exp.dp_cfg["dataset"], "coll": f"{src}-{search}-{data_split}"}),  # type: ignore
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        # Load models
        exp.log(f"Load model with mid {mid}...")
        model_root = os.path.join(exp.exp_dump_path, "models")
        models = []
        if seeds is not None:
            model_files = [f"model-{mid}_seed{seed}.pth" for seed in seeds]
        else:
            model_files = [f"model-{mid}.pth"]
        for model_file in sorted(model_files):
            model = build_model(args.model_name, exp.model_params)
            model.load_state_dict(torch.load(os.path.join(model_root, model_file), map_location=DEVICE))
            model.to(DEVICE)
            models.append(model)

        if data_split != "test":
            exp.log(f"== Run Evaluation on {data_split.upper()} Set Using Ckpt {mid} ==")

            evaluator = Evaluator(metric_names=["opa", "kendall_tau"])
            y_true, y_pred = _run_infer(models, dataloader)
            n_samples = len(y_pred)
            eval_result = defaultdict(float)  # type: ignore
            for i, (y_pred_i, y_true_i) in enumerate(zip(y_pred, y_true)):
                eval_result_i = evaluator.evaluate(y_pred_i.reshape(1, -1), y_true_i.reshape(1, -1), None)
                for eval_metric, score in eval_result_i.items():
                    eval_result[eval_metric] += score / n_samples
            exp.log(f">>>>> Evaluation Result Using Ckpt {mid} <<<<<")
            exp.log(json.dumps(eval_result, indent=4))
        else:
            exp.log(f"== Run Inference on {data_split.upper()} Set Using Ckpt {mid} ==")
            y_true, y_pred = _run_infer(models, dataloader)
            top_configs = []
            for y_pred_g in y_pred:
                top_configs_g = np.argsort(y_pred_g.numpy())
                top_configs.append(";".join(top_configs_g.astype(str)))

            sub = _gen_sub(data, top_configs, src, search)
            exp.dump_df(sub, "submission.csv")


if __name__ == "__main__":
    # Parse arguments
    arg_parser = InferArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
