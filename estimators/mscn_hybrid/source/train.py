import math
import torch
import torch.nn.functional as F

from typing import Dict, Tuple
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, AdamW
from common.logger import Logger
from common.torch_utils import normalize_card, unnormalize_card, q_error
from estimators.mscn_hybrid.source.model import SetConvEnsemble


def gaussian_nll_loss(mu_all: torch.Tensor, logvar_all: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood loss for ensemble heads.

    Args:
        mu_all:      [batch_size, ensemble_size] — predicted means (normalized)
        logvar_all:  [batch_size, ensemble_size] — predicted log-variances
        target:      [batch_size] — ground truth (normalized)

    Returns:
        Scalar NLL loss averaged over all heads and samples.
    """
    target_expanded = target.unsqueeze(1).expand_as(mu_all)
    var_all = torch.exp(logvar_all) + 1e-8

    squared_error = (target_expanded - mu_all) ** 2
    # ℒ = ½(log σ² + (y-μ)²/σ² + log(2π))
    nll = 0.5 * (logvar_all + squared_error / var_all + math.log(2 * math.pi))

    return nll.mean()


class MSCNHybridTrainer:
    def __init__(self, model: SetConvEnsemble, device: str, log: Logger, config: Dict) -> None:
        self._model = model
        self._device = device
        self._log = log
        self._batch_size = config["batch_size"]
        self._epochs = config["epochs"]

        if config["optimizer"] == "adam":
            self._optimizer = Adam(model.parameters(), lr=config["lr"])
        elif config["optimizer"] == "adamw":
            self._optimizer = AdamW(model.parameters(), lr=config["lr"])
        else:
            raise NotImplementedError

        self._lr = config["lr"]

        self._early_stop_patience = (
            config["early_stop_patience"]
            if config["early_stop_patience"] is not None
            else float("inf")
        )

        self._not_improved_count = 0
        self._best_valid_loss = float("inf")
        self._best_model = None

    def train(
        self,
        train_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor, torch.Tensor],
        train_target: torch.Tensor,
        valid_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor, torch.Tensor],
        valid_target: torch.Tensor,
    ) -> None:
        # Normalize targets to [0, 1] scale
        train_target_norm = normalize_card(train_target)
        valid_target_norm = normalize_card(valid_target)

        train_dataset = TensorDataset(*train_input, train_target_norm)
        valid_dataset = TensorDataset(*valid_input, valid_target_norm)
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self._batch_size, shuffle=True)

        self._model.to(self._device)
        for epoch in range(self._epochs):
            self._train_epoch(train_loader, epoch + 1)
            avg_loss = self._valid_epoch(valid_loader, epoch + 1)

            if avg_loss >= self._best_valid_loss:
                self._not_improved_count += 1
            else:
                self._not_improved_count = 0
                self._best_valid_loss = avg_loss
                self._best_model = {
                    k: v.detach().cpu().clone()
                    for k, v in self._model.state_dict().items()
                }
                self._log.write(f"New best model on epoch {epoch + 1}! "
                                f"NLL Loss: {avg_loss:.6f}\n")

            if self._not_improved_count >= self._early_stop_patience:
                break

    def _train_epoch(self, loader: DataLoader, epoch_idx: int) -> None:
        self._model.train()
        self._log.write(f"Epoch {epoch_idx}\n")

        for batch_idx, (samples, predicates, joins, sample_mask,
                        predicate_mask, join_mask, target) in enumerate(loader):
            samples = samples.to(self._device)
            predicates = predicates.to(self._device)
            joins = joins.to(self._device)
            sample_mask = sample_mask.to(self._device)
            predicate_mask = predicate_mask.to(self._device)
            join_mask = join_mask.to(self._device)
            target = target.to(self._device)

            self._optimizer.zero_grad()
            mu_all, logvar_all = self._model(samples, predicates, joins,
                                             sample_mask, predicate_mask, join_mask)
            loss = gaussian_nll_loss(mu_all, logvar_all, target)
            loss.backward()
            self._optimizer.step()

            if batch_idx % 100 == 0:
                self._log.write(f"Batch {batch_idx + 1}: NLL loss = {loss.item():.6f}\n")

    def _valid_epoch(self, loader: DataLoader, epoch_idx: int) -> float:
        self._model.eval()
        self._log.write(f"Validating epoch {epoch_idx}\n")

        with torch.no_grad():
            total_loss = 0.0
            total_samples = 0

            for batch_idx, (samples, predicates, joins, sample_mask,
                            predicate_mask, join_mask, target) in enumerate(loader):
                samples = samples.to(self._device)
                predicates = predicates.to(self._device)
                joins = joins.to(self._device)
                sample_mask = sample_mask.to(self._device)
                predicate_mask = predicate_mask.to(self._device)
                join_mask = join_mask.to(self._device)
                target = target.to(self._device)

                mu_all, logvar_all = self._model(samples, predicates, joins,
                                                 sample_mask, predicate_mask, join_mask)
                loss = gaussian_nll_loss(mu_all, logvar_all, target)

                # Also compute ensemble q-error for reference
                mu_ensemble = mu_all.mean(dim=1)
                pred_raw = unnormalize_card(mu_ensemble)
                true_raw = unnormalize_card(target)
                qe = q_error(pred_raw, true_raw).item()

                self._log.write(f"Validation Batch {batch_idx}: "
                                f"NLL = {loss.item():.6f}, "
                                f"Ensemble Q-error = {qe:.4f}\n")

                total_loss += float(loss.item()) * len(target)
                total_samples += len(target)

        avg_loss = total_loss / total_samples
        self._log.write(f"Average validation NLL loss: {avg_loss:.6f}\n")

        return avg_loss
