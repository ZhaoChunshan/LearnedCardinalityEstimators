import math
import torch


from typing import Dict, Tuple
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, AdamW
from estimators.alece.source.model import ALECE
from common.logger import Logger


class ALECETrainer:
    def __init__(self, model: ALECE, device: str, log: Logger, config: Dict) -> None:
        self._model = model
        self._device = device
        self._log = log
        self._batch_size = config["batch_size"]
        self._epochs = config["epochs"]

        if config["optimizer"] == "adam":
            self._optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        elif config["optimizer"] == "adamw":
            self._optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        else:
            raise NotImplementedError

        self._lr = config["lr"]
        self._weight_decay = config["weight_decay"]

        self._early_stop_patience = config["early_stop_patience"] if config["early_stop_patience"] is not None else float("inf")

        self._best_valid_loss = float("inf")
        self._best_model = None
        self._not_improved_count = 0


    def train(
            self,
            train_input: Tuple[torch.Tensor, torch.Tensor], # x, q
            train_target: torch.Tensor,                     # log(card).
            train_weight: torch.Tensor,                     # weight
            valid_input: Tuple[torch.Tensor, torch.Tensor], # x, q
            valid_target: torch.Tensor,                     # log(card).
            valid_weight: torch.Tensor,                     # weight
            ) -> None:
        train_dateset = TensorDataset(*train_input, train_target, train_weight)
        valid_dataset = TensorDataset(*valid_input, valid_target, valid_weight)
        train_loader = DataLoader(train_dateset, batch_size=self._batch_size, shuffle=True)
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
            
    def train_wo_valid(
            self,
            train_input: Tuple[torch.Tensor, torch.Tensor], # x, q
            train_target: torch.Tensor,                     # log(card).
            train_weight: torch.Tensor,                     # weight
            ) -> None:
        train_dateset = TensorDataset(*train_input, train_target, train_weight)
        train_loader = DataLoader(train_dateset, batch_size=self._batch_size, shuffle=True)

        self._model.to(self._device)
        for epoch in range(self._epochs):
            self._train_epoch(train_loader, epoch + 1)

            if epoch == self._epochs - 1:
                self._best_model = {
                    k: v.detach().cpu().clone()
                    for k, v in self._model.state_dict().items()
                }
            
        
    def _train_epoch(self, loader: DataLoader, epoch_idx: int) -> None:
        self._model.train()
        self._log.write(f"Epoch {epoch_idx}\n")
        for batch_idx, (x, q, target, weight) in enumerate(loader):
            x = x.to(self._device)
            q = q.to(self._device)
            target = target.to(self._device)
            weight = weight.to(self._device)

            self._optimizer.zero_grad()
            output = self._model(x, q)
            loss = torch.mean(weight * (output - target) ** 2)
            loss.backward()
            self._optimizer.step()
            if batch_idx % 100 == 0:
                self._log.write(f"Batch {batch_idx + 1}: loss = {loss.item()}\n")
            

    def _valid_epoch(self, loader: DataLoader, epoch_idx: int) -> None:
        self._model.eval()
        self._log.write(f"Validating epoch {epoch_idx}\n")
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            for batch_idx, (x, q, target, weight) in enumerate(loader):
                x = x.to(self._device)
                q = q.to(self._device)
                target = target.to(self._device)
                weight = weight.to(self._device)

                output = self._model(x, q)
                loss = torch.mean(weight * (output - target) ** 2)
                self._log.write(f"Validation Batch {batch_idx}: loss = {loss.item()}\n")
                total_loss += float(loss.item()) * target.shape[0]
                total_samples += target.shape[0]
        
        avg_loss = total_loss / total_samples

        self._log.write(f"Average loss: {avg_loss}\n")

        return avg_loss
