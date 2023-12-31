"""
Main script for training and evaluation processes.
Author: JiaWei Jiang

This script is used to run training and evaluation processes given the
specified CV scheme. Moreover, evaluation on unseen (test) data is
optional.

* [ ] Extend to multiple folds?
"""
import gc
import math
import os
import warnings
from argparse import Namespace

import pandas as pd
import torch
import wandb

from base.base_trainer import BaseTrainer
from criterion.build import build_criterion
from data.build import build_layout_dataloaders
from engine.defaults import TrainEvalArgParser
from evaluating.build import build_evaluator
from experiment.experiment import Experiment
from modeling.build import build_model
from solver.build import build_lr_scheduler, build_optimizer
from trainer.trainer import GSTrainer
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
        coll = exp.dp_cfg["coll"]

        # Prepare data
        src, search = coll.split("-")
        data = pd.read_pickle(f"./data/processed/layout/{src}/{search}/train_new_206.pkl")
        fold_col = "strat_f5_s42_y_mean_log"
        data_tr, data_val = data[data[fold_col] != 3].reset_index(drop=True), data[data[fold_col] == 3].reset_index(
            drop=True
        )

        # Run CV (support only train/val split now)
        for fold in range(1):
            # Configure sub-entry for tracking current fold
            if args.use_wandb:
                sub_entry = wandb.init(
                    project=args.project_name,
                    group=exp.exp_id,
                    job_type="train_eval",
                    # name=f"fold{fold}",
                    name=f"seed{args.seed_num}",
                )
            # exp.log(f"\n== Train and Eval Process - Fold{fold} ==")
            exp.log(f"\n== Train and Eval Process - Seed{args.seed_num} ==")

            # Build dataloaders
            train_loader, val_loader = build_layout_dataloaders(
                data_tr,
                data_val,
                coll,
                test=False,
                **exp.proc_cfg["dataloader"],
                **exp.dp_cfg["dataset"],
            )

            # Build model
            model = build_model(args.model_name, exp.model_params)
            if args.pretrained_path is not None:
                exp.log(f"Load model weights from pretrained ckpt {args.pretrained_path}...")
                model.load_state_dict(torch.load(args.pretrained_path, map_location=exp.proc_cfg["device"]))
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

            # Run main training and evaluation for one fold
            trainer.train_eval(fold)

            # Run evaluation on unseen test set
            if args.eval_on_test:
                # data_test = dp.get_data_test()
                _, test_loader = build_layout_dataloaders(
                    coll,
                    test=True,
                    **exp.proc_cfg["dataloader"],
                    **exp.dp_cfg["dataset"],
                )
                _ = trainer.test(fold, test_loader)

            trainer.profiler.summarize(log_wnb=True if args.use_wandb else False)

            # Dump output objects
            # Use args to control whether to dump the preds of diff datasets
            # if scalers is not None:
            # exp.dump_trafo(scalers, f"fold{fold}")
            for model_file in os.listdir(ckpt_path):
                # if "fold" in model_file:
                if "seed" in model_file:
                    continue

                model_file_name = model_file.split(".")[0]
                model_path_src = os.path.join(ckpt_path, model_file)
                # model_path_dst = os.path.join(ckpt_path, f"{model_file_name}_fold{fold}.pth")
                model_path_dst = os.path.join(ckpt_path, f"{model_file_name}_seed{args.seed_num}.pth")
                os.rename(model_path_src, model_path_dst)

            # Free mem.
            del (
                # data_tr,
                # data_val,
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
