"""
Model architecture building logic.
Author: JiaWei Jiang

The file contains a single function for model name switching and model
architecture building based on the model configuration.

To add in new model, users need to design custom model architecture,
put the file under the same directory, and import the corresponding
model below (i.e., import the separately pre-defined model arch.).
"""
from typing import Any, Dict

from torch.nn import Module

from .baseline.Baseline import EarlyJoinGConv, LateJoinGAT, LateJoinGConv
from .baseline.LayoutBaseline import LayoutEarlyJoinGConv
from .exp.Exp2 import Exp


def build_model(model_name: str, model_cfg: Dict[str, Any]) -> Module:
    """Build and return the specified model architecture.

    Parameters:
        model_name: name of model architecture
        model_cfg: hyperparameters of the specified model

    Return:
        model: model instance
    """
    model: Module
    if model_name == "EarlyJoinGConv":
        model = EarlyJoinGConv(**model_cfg)
    elif model_name == "LateJoinGConv":
        model = LateJoinGConv(**model_cfg)
    elif model_name == "LateJoinGAT":
        model = LateJoinGAT(**model_cfg)
    elif model_name == "LayoutEarlyJoinGConv":
        model = LayoutEarlyJoinGConv(**model_cfg)
    elif model_name == "Exp2":
        model = Exp(**model_cfg)
    else:
        raise RuntimeError(f"{model_name} isn't registered.")

    return model
