"""
Evaluator building logic.
Author: JiaWei Jiang

This file contains the basic logic of building evaluator for evaluation
process.
"""
from typing import List

from .evaluator import Evaluator


def build_evaluator(
    eval_metrics: List[str],
) -> Evaluator:
    """Build and return the evaluator.

    Parameters:
        metric_names: evaluation metrics

    Return:
        evaluator: evaluator
    """
    evaluator = Evaluator(eval_metrics)

    return evaluator
