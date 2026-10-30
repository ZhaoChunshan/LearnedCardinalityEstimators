import torch
import torch.nn.functional as F

from common.constants import MAX_CARDINALITY, MAX_LOG_CARDINALITY
from typing import Any


def create_one_hot(l: int, i: int) -> torch.Tensor:
    tensor = torch.zeros(l)
    tensor[i] = 1
    return tensor


def normalize_card(cardinality: torch.Tensor):
    return torch.log(cardinality + 1) / MAX_LOG_CARDINALITY


def unnormalize_card(cardinality: torch.Tensor):
    return torch.exp(cardinality * MAX_LOG_CARDINALITY) - 1


def q_error(y_pred: torch.Tensor, y_true: torch.Tensor):
    """
        Assume y_pred and y_true are non-negative
    """
    return torch.max(y_pred/y_true, y_true/y_pred).mean()


def q_error_smooth(y_pred: torch.Tensor, y_true: torch.Tensor):
    pred = torch.clamp(y_pred, min=1.0)
    true = torch.clamp(y_true, min=1.0)

    return torch.max(pred/true, true/pred).mean()


def log_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    return F.mse_loss(torch.log(y_pred + 1), torch.log(y_true + 1))
