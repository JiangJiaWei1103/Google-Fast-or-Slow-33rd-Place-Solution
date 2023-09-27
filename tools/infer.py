"""
Main script for inference processes.
Author: JiaWei Jiang
"""
import gc
import json
import logging
import os
import warnings
from argparse import Namespace
from typing import List

import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from base.base_argparser import BaseArgParser
from data.build import build_dataloaders
from evaluating.build import build_evaluator
from experiment.experiment import Experiment
from modeling.build import build_model

warnings.simplefilter("ignore")
DEVICE = torch.device("cuda:0")


class InferArgParser(BaseArgParser):
    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        self.argparser.add_argument("--exp-id", type=str, default=None)
        self.argparser.add_argument("--data-split", type=str, default="valid", choices=["valid", "test"])
        self.argparser.add_argument("--model-name", type=str, default=None)
        self.argparser.add_argument("--mid", type=str, default=None)
        self.argparser.add_argument("--use-wandb", type=self._str2bool, default=False)


def _run_infer(models: List[Module], dataloader: DataLoader) -> List[Tensor]:
    """Run inference using well-trained models."""
    y_pred = []
    n_models = len(models)

    logging.info(f"\tStart batchified inference with {n_models} models...")
    models = [model.eval() for model in models]
    for i, batch_data in enumerate(dataloader):
        # Retrieve batched raw data
        inputs = {}
        for k, v in batch_data.items():
            if k != "target":
                inputs[k] = v.to(DEVICE)

        # Forward pass
        for j, model in enumerate(models):
            if j == 0:
                y_pred_i = model(inputs) / n_models
            else:
                y_pred_i += model(inputs) / n_models

        # Record batched output
        y_pred.append(y_pred_i.detach().cpu())

        del inputs, y_pred_i
        _ = gc.collect()

    return y_pred


def main(args: Namespace) -> None:
    """Run training and evaluation processes.

    Parameters:
        args: arguments driving training and evaluation processes

    Returns:
        None
    """
    # Configure experiment
    experiment = Experiment(args, log_file="infer.log", infer=True)
    data_split = args.data_split
    mid = args.mid

    with experiment as exp:
        # Prepare data
        data = pd.read_pickle(f"./data/processed/tile/xla/{data_split}.pkl")
        dataloader, _ = build_dataloaders(
            data, None, batch_size=256, shuffle=False, num_workers=4, **exp.dp_cfg["dataset"]
        )

        # Load models
        model_root = os.path.join(exp.exp_dump_path, "models")
        models = []
        for model_file in sorted(os.listdir(model_root)):
            if mid not in model_file:
                continue

            model = build_model(args.model_name, exp.model_params)
            model.load_state_dict(torch.load(os.path.join(model_root, model_file), map_location=DEVICE))
            model.to(DEVICE)
            models.append(model)

        # Run inference
        exp.log(f"== Inference Process on {data_split.upper()} ==")
        y_pred = _run_infer(models, dataloader)

        # Run evaluation
        if data_split != "test":
            y_true = dataloader.dataset.data["target"].values
            y_pred = torch.cat(y_pred, dim=-1)
            evaluator = build_evaluator(**exp.proc_cfg["evaluator"])
            eval_result = evaluator.evaluate(y_true, y_pred, None)
            exp.log(">>>>> Performance Report - Ckpt {mid} <<<<<")
            exp.log(json.dumps({data_split: eval_result}, indent=4))
        else:
            # Structure prediction to fit submission
            pass


if __name__ == "__main__":
    # Parse arguments
    arg_parser = InferArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
