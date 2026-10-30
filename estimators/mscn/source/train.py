import math
import torch


from typing import Dict, Tuple
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, AdamW
from common.logger import Logger
from common.torch_utils import q_error, unnormalize_card
from estimators.mscn.source.model import SetConv


class MSCNTrainer:
    def __init__(self, model: SetConv, device: str, log: Logger, config: Dict) -> None:
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

        self._early_stop_patience = config["early_stop_patience"] if config["early_stop_patience"] is not None else float("inf")

        self._not_improved_count = 0
        
        self._best_valid_loss = float("inf")
        self._best_model = None

    def train(self, train_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], train_target: torch.Tensor, 
              valid_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], valid_target: torch.Tensor) -> None:
        train_dataset = TensorDataset(*train_input, train_target)
        valid_dataset = TensorDataset(*valid_input, valid_target)
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
                self._log.write(f"New best model on epoch {epoch + 1}! Loss: {avg_loss:.4f}\n")

            if self._not_improved_count >= self._early_stop_patience:
                break

    def _train_epoch(self, loader: DataLoader, epoch_idx: int) -> None:
        self._model.train()
        self._log.write(f"Epoch {epoch_idx}\n")

        for batch_idx, (samples, predicates, joins, sample_mask, predicate_mask, join_mask, target) in enumerate(loader):
            samples = samples.to(self._device)
            predicates = predicates.to(self._device)
            joins = joins.to(self._device)
            sample_mask = sample_mask.to(self._device)
            predicate_mask = predicate_mask.to(self._device)
            join_mask = join_mask.to(self._device)
            target = target.to(self._device)

            self._optimizer.zero_grad()
            output = self._model(samples, predicates, joins, sample_mask, predicate_mask, join_mask).view(-1)
            output = unnormalize_card(output)
            loss = q_error(output, target)
            loss.backward()
            self._optimizer.step()
            if batch_idx % 100 == 0:
                self._log.write(f"Batch {batch_idx + 1}: loss = {loss.item()}\n")

    def _valid_epoch(self, loader: DataLoader, epoch_idx: int) -> float:
        self._model.eval()
        self._log.write(f"Validating epoch {epoch_idx}\n")

        with torch.no_grad():
            total_loss = 0
            total_samples = 0

            for batch_idx, (samples, predicates, joins, sample_mask, predicate_mask, join_mask, target) in enumerate(loader):
                samples = samples.to(self._device)
                predicates = predicates.to(self._device)
                joins = joins.to(self._device)
                sample_mask = sample_mask.to(self._device)
                predicate_mask = predicate_mask.to(self._device)
                join_mask = join_mask.to(self._device)
                target = target.to(self._device)

                output = self._model(samples, predicates, joins, sample_mask, predicate_mask, join_mask).view(-1)
                output = unnormalize_card(output)
                loss = q_error(output, target)

                self._log.write(f"Validation Batch {batch_idx}: loss = {loss.item()}\n")

                total_loss += float(loss.item()) * len(target)
                total_samples += len(target)

        avg_loss = total_loss / total_samples

        self._log.write(f"Average loss: {avg_loss}\n")

        return avg_loss
