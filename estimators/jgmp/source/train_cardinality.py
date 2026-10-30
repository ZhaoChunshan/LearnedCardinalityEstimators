import math
import torch

from torch.optim import Adam, AdamW
from torch_geometric.loader.dataloader import DataLoader
from typing import Dict
from common.logger import Logger


from estimators.jgmp.source.models.bulk_jgmp_cardinality_model import BulkJGMPCardinalityModel
from estimators.jgmp.source.query_data.bulk_light_cardinality_query_data import BulkLightCardinalityQueryData


class JGMPTrainer:
    def __init__(self, model: BulkJGMPCardinalityModel, device: str, log: Logger, config: Dict) -> None:
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

        self._best_valid_loss = float("inf")
        self._best_model = None
        self._not_improved_count = 0

    def train(self, train_data: BulkLightCardinalityQueryData, valid_data: BulkLightCardinalityQueryData) -> None:
        train_loader = DataLoader(train_data, batch_size=self._batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=self._batch_size, shuffle=True)

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

        for batch_idx, data in enumerate(loader):
            data = data.to(self._device)
            self._optimizer.zero_grad()
            loss = self._model.loss(data)
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

            for batch_idx, data in enumerate(loader):
                data = data.to(self._device)
                loss = self._model.loss(data)

                self._log.write(f"Validation Batch {batch_idx}: loss = {loss.item()}\n")

                total_loss += float(loss.item()) * len(data)
                total_samples += len(data)

            avg_loss = total_loss / total_samples

            self._log.write(f"Average loss: {avg_loss}\n")

            return avg_loss
