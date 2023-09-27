"""
Main script for training and evaluation processes.
Author: JiaWei Jiang

This script is used to run training and evaluation processes given the
specified CV scheme. Moreover, evaluation on unseen (test) data is
optional.

* [ ] Add wandb tracker via public method in `Experiment` (e.g.,
    `add_wnb_entry` method).
* [ ] Feed priori graph structure when calling model `forward`.
"""
import gc
import math
import os
import warnings
from argparse import Namespace

import pandas as pd
from transformers import get_cosine_schedule_with_warmup

import wandb
from base.base_trainer import BaseTrainer
from criterion.build import build_criterion
from cv.build import build_cv
from data.build import build_dataloaders
from engine.defaults import TrainEvalArgParser
from evaluating.build import build_evaluator
from experiment.experiment import Experiment
from modeling.build import build_model
from solver.build import build_optimizer
from trainer.trainer import MainTrainer
from utils.common import count_params
from utils.early_stopping import EarlyStopping

warnings.simplefilter("ignore")


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

        # Prepare data
        data = pd.read_pickle("./data/processed/tile/xla/train.pkl")

        # Run CV
        cv = build_cv(**exp.dp_cfg["cv"])
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X=data)):
            # Configure sub-entry for tracking current fold
            if args.use_wandb:
                sub_entry = wandb.init(
                    project=args.project_name,
                    group=exp.exp_id,
                    job_type="train_eval",
                    name=f"fold{fold}",
                )
            exp.log(f"\n== Train and Eval Process - Fold{fold} ==")

            # Build dataloaders
            data_tr, data_val = data.iloc[tr_idx].reset_index(drop=True), data.iloc[val_idx].reset_index(drop=True)
            scaler = None
            train_loader, val_loader = build_dataloaders(
                data_tr,
                data_val,
                **exp.proc_cfg["dataloader"],
                **exp.dp_cfg["dataset"],
            )

            # Build model
            model = build_model(args.model_name, exp.model_params)
            model.to(exp.proc_cfg["device"])
            if args.use_wandb:
                wandb.log({"model": {"n_params": count_params(model)}})
                wandb.watch(model, log="all", log_graph=True)

            # Build criterion
            loss_fn = build_criterion(**exp.proc_cfg["loss_fn"])

            # Build solvers
            optimizer = build_optimizer(model, **exp.proc_cfg)
            n_train_steps = math.ceil(len(data_tr) / exp.proc_cfg["dataloader"]["batch_size"]) * exp.proc_cfg["epochs"]
            lr_skd = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_train_steps)
            # lr_skd = build_lr_scheduler(optimizer, **exp.proc_cfg)

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
                "scaler": scaler,
                "use_wandb": args.use_wandb,
            }
            trainer = MainTrainer(**trainer_cfg)

            # Run main training and evaluation for one fold
            trainer.train_eval(fold)

            # Run evaluation on unseen test set
            if args.eval_on_test:
                # data_test = dp.get_data_test()
                data_test = pd.read_pickle("./data/processed/tile/xla/valid.pkl")
                _, test_loader = build_dataloaders(
                    data_tr,
                    data_test,
                    **exp.proc_cfg["dataloader"],
                    **exp.dp_cfg["dataset"],
                )
                _ = trainer.test(fold, test_loader)

            trainer.profiler.summarize(log_wnb=True if args.use_wandb else False)

            # Dump output objects
            # Use args to control whether to dump the preds of diff datasets
            # exp.dump_trafo(scaler, f"fold{i}")
            for model_file in os.listdir(ckpt_path):
                if "fold" in model_file:
                    continue

                model_file_name = model_file.split(".")[0]
                model_path_src = os.path.join(ckpt_path, model_file)
                model_path_dst = os.path.join(ckpt_path, f"{model_file_name}_fold{fold}.pth")
                os.rename(model_path_src, model_path_dst)

            # Free mem.
            del (
                data_tr,
                data_val,
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
    arg_parser = TrainEvalArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
