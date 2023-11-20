"""
Main script for full-training process.
Author: JiaWei Jiang
"""
import gc
import math
import os
import random
import warnings
from argparse import Namespace

import pandas as pd
from torch_geometric.loader.dataloader import DataLoader  # noqa

import wandb
from base.base_trainer import BaseTrainer
from config.config import _seed_all
from criterion.build import build_criterion
from data.dataset import LayoutDataset
from engine.defaults import TrainEvalArgParser
from evaluating.build import build_evaluator
from experiment.experiment import Experiment
from modeling.build import build_model
from solver.build import build_lr_scheduler, build_optimizer
from trainer.trainer import GSTrainer
from utils.common import count_params
from utils.early_stopping import EarlyStopping

warnings.simplefilter("ignore")


class TrainAllArgParser(TrainEvalArgParser):
    def __init__(self) -> None:
        super().__init__()

        self.argparser.add_argument("--n-seeds", type=int, default=5)


def main(args: Namespace) -> None:
    """Run training and evaluation processes.

    Parameters:
        args: arguments driving training and evaluation processes

    Returns:
        None
    """
    # Configure experiment
    experiment = Experiment(args)

    with experiment as exp:
        exp.dump_cfg(exp.cfg, "cfg")
        coll = exp.dp_cfg["coll"]

        # Prepare data
        src, search = coll.split("-")
        data = pd.read_pickle(f"./data/processed/layout/{src}/{search}/train_new_206.pkl")

        seeds = [random.randint(1, 2**32 - 1) for _ in range(args.n_seeds)]
        for s_i, seed in enumerate(seeds):
            _seed_all(seed)

            # Configure sub-entry for tracking current fold
            if args.use_wandb:
                sub_entry = wandb.init(
                    project=args.project_name,
                    group=exp.exp_id,
                    job_type="full_train",
                    name=f"seed{s_i}",
                )
            exp.log(f"\n== Full-Training Process - Seed{s_i} ==")

            # Build dataloaders
            train_loader = DataLoader(
                LayoutDataset(data, **{**exp.dp_cfg["dataset"], "coll": f"{coll}-train"}),  # type: ignore
                batch_size=exp.proc_cfg["dataloader"]["batch_size"],
                shuffle=exp.proc_cfg["dataloader"]["shuffle"],
                num_workers=exp.proc_cfg["dataloader"]["num_workers"],
            )
            val_loader = None

            # Build model
            model = build_model(args.model_name, exp.model_params)
            model.to(exp.proc_cfg["device"])
            if args.use_wandb:
                wandb.log({"model": {"n_params": count_params(model)}})
                wandb.watch(model, log="all", log_graph=True)

            # Build criterion
            loss_fn = build_criterion(**exp.proc_cfg["loss_fn"])

            # Build solvers
            optimizer = build_optimizer(model, **exp.proc_cfg["solver"]["optimizer"])
            num_training_steps = (
                math.ceil(
                    len(train_loader.dataset)
                    / exp.proc_cfg["dataloader"]["batch_size"]
                    / exp.proc_cfg["solver"]["optimizer"]["grad_accum_steps"]
                )
                * exp.proc_cfg["epochs"]
            )
            lr_skd = build_lr_scheduler(optimizer, num_training_steps, **exp.proc_cfg["solver"]["lr_skd"])

            # Build early stopping tracker
            if exp.proc_cfg["es"]["patience"] != 0:
                es = EarlyStopping(exp.proc_cfg["es"]["patience"], exp.proc_cfg["es"]["mode"])
            else:
                es = None

            # Build evaluator
            evaluator = build_evaluator(**exp.proc_cfg["evaluator"])

            # Build trainer
            ckpt_path = os.path.join(exp.exp_dump_path, "models")
            trainer: BaseTrainer = None
            trainer_cfg = {
                "logger": exp.logger,
                "proc_cfg": exp.proc_cfg,
                "model": model,
                "loss_fn": loss_fn,
                "optimizer": optimizer,
                "lr_skd": lr_skd,
                "ckpt_path": ckpt_path,
                "es": es,
                "evaluator": evaluator,
                "train_loader": train_loader,
                "eval_loader": val_loader,
                "scaler": None,  # scaler,
                "use_wandb": args.use_wandb,
            }
            trainer = GSTrainer(**trainer_cfg)

            # Run full-training process for one seed
            trainer.train_only()
            trainer.profiler.summarize(log_wnb=True if args.use_wandb else False)

            # Dump output objects
            for model_file in os.listdir(ckpt_path):
                if "seed" in model_file:
                    continue

                model_file_name = model_file.split(".")[0]
                model_path_src = os.path.join(ckpt_path, model_file)
                model_path_dst = os.path.join(ckpt_path, f"{model_file_name}_seed{s_i}.pth")
                os.rename(model_path_src, model_path_dst)

            # Free mem.
            del (
                train_loader,
                val_loader,
                model,
                optimizer,
                lr_skd,
                es,
                evaluator,
                trainer,
            )
            _ = gc.collect()

            if args.use_wandb:
                sub_entry.finish()


if __name__ == "__main__":
    # Parse arguments
    arg_parser = TrainAllArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
