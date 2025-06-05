# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from typing import Dict, Tuple
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from transformers import EvalPrediction

import logging
import torch

# set up logging basic config from info
logging.basicConfig(level=logging.INFO)


def compute_metrics_for_prob_nlabel(
    eval_pred: EvalPrediction,
) -> Dict[str, Dict[str, float]]:

    pred_scores, labels = eval_pred

    scores = labels[..., 0]
    agg_scores = np.mean(scores, axis=-1)

    pred_diff_list = []
    agg_score_diff_list = []
    for i, p in enumerate(pred_scores):
        pred_diff_list.append([_[1] - _[0] for _ in p])
        agg_score_diff_list.append([_[-1] - _[i] for _ in agg_scores])

    pred_diff_list = np.array(pred_diff_list)
    agg_score_diff_list = np.array(agg_score_diff_list)

    spearman_score = []
    for i in range(len(pred_diff_list)):
        spearman_score.append(spearmanr(pred_diff_list[i], agg_score_diff_list[i]).correlation)

    metrics = {
        "sel": {'spearman_score': spearman_score},
        "oracle": {},
        "dev_score": np.mean(spearman_score),
    }

    return metrics


def compute_metrics_for_det_nclass(
    eval_pred: EvalPrediction,
) -> Dict[str, Dict[str, float]]:

    pred_scores, labels = eval_pred
    scores = labels[..., 0]
    agg_scores = np.mean(scores, axis=-1)

    pred_scores = torch.argmax(torch.tensor(pred_scores), dim=-1)
    true_labels = torch.argmax(torch.tensor(agg_scores), dim=-1)

    metrics = {
        "sel": {},
        "oracle": {},
        "dev_score": accuracy_score(true_labels, pred_scores),
    }

    return metrics

def compute_metrics_for_det_nlabel(
    eval_pred: EvalPrediction,
) -> Dict[str, Dict[str, float]]:

    pred_scores, labels = eval_pred
    scores = labels[..., 0]
    agg_scores = np.mean(scores, axis=-1)

    pred_scores = torch.sigmoid(torch.tensor(pred_scores))
    true_labels = torch.tensor([[(_ >= s[-1]) * 1 for _ in s] for s in agg_scores])

    metrics = {
        "sel": {},
        "oracle": {},
        "dev_score": torch.nn.functional.mse_loss(pred_scores, true_labels).item(),
    }

    return metrics


def compute_metrics_for_prob_2class(
    eval_pred: EvalPrediction,
) -> Dict[str, Dict[str, float]]:
    """
    Computes various metrics for n-class classification.

    Args:
        eval_pred (EvalPrediction): The evaluation predictions containing the predicted scores and labels.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the computed metrics.
            - "sel":
                - "avg_score": The average score of the selected pairs.
                - "acc": The accuracy of the selected pairs.
                - "avg_ind": The average index of the selected pairs.
            - "oracle":
                - "avg_score": The average score of the oracle-selected pairs.
                - "acc": The accuracy of the oracle-selected pairs.
                - "avg_ind": The average index of the oracle-selected pairs.
            - "dev_score": The Spearman correlation coefficient between the predicted scores and aggregated scores.
    """
    pred_scores, labels = eval_pred

    scores = labels[..., 0]
    agg_scores = np.mean(scores, axis=-1)

    pred_diff_list = []
    agg_score_diff_list = []
    for p, s in zip(pred_scores, agg_scores):
        pred_diff_list += [p[-1] - _ for _ in p[:-1]]
        agg_score_diff_list += [s[-1] - _ for _ in s[:-1]]

    spearman_score = spearmanr(
        np.array(pred_diff_list), np.array(agg_score_diff_list)
    ).correlation

    ranks = calculate_ranks(agg_scores)
    sel_scores, sel_acc, sel_idx = calculate_selections(scores, ranks, pred_scores)
    oracle_sel_scores, oracle_sel_acc, oracle_sel_idx = calculate_selections(
        scores, ranks, agg_scores
    )

    metrics = {
        "sel": {
            "avg_score": np.mean(sel_scores),
            "acc": sel_acc,
            "avg_ind": np.mean(sel_idx),
        },
        "oracle": {
            "avg_score": np.mean(oracle_sel_scores),
            "acc": oracle_sel_acc,
            "avg_ind": np.mean(oracle_sel_idx),
        },
        "dev_score": spearman_score,
    }

    return metrics


def calculate_ranks(scores: np.ndarray) -> np.ndarray:
    """
    Calculate ranks based on the given scores.

    Parameters:
    scores (numpy.ndarray): The scores used to calculate the ranks.

    Returns:
    numpy.ndarray: The calculated ranks.

    """
    sort_indices = np.flip(np.argsort(scores, axis=-1), axis=-1)
    ranks = np.zeros_like(sort_indices)
    ranks[np.arange(sort_indices.shape[0])[:, None], sort_indices] = np.arange(
        sort_indices.shape[-1]
    )
    return ranks


def calculate_selections(
    scores: np.ndarray, ranks: np.ndarray, pred_scores: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Calculates the selections based on the given scores, ranks, and predicted scores.

    Parameters:
    scores (numpy.ndarray): Array of shape (n, m) representing the scores.
    ranks (numpy.ndarray): Array of shape (n, m) representing the ranks.
    pred_scores (numpy.ndarray): Array of shape (n, k) representing the predicted scores.

    Returns:
    tuple: A tuple containing the selected scores, accuracy, and indices.

    """
    sel_idx = np.argmax(pred_scores, axis=1)
    sel_scores = scores[np.arange(scores.shape[0]), sel_idx]
    sel_ranks = ranks[np.arange(ranks.shape[0]), sel_idx]
    sel_acc = np.mean((sel_ranks == 0))
    return sel_scores, sel_acc, sel_idx
