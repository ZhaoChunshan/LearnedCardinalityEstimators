import math
import torch

from typing import Dict, Tuple, Optional
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, AdamW, lr_scheduler
from estimators.price.source.model.encoder import RegressionModel
from estimators.price.source.utils.dataset import CustomDataset
from common.logger import Logger


class PRICETrainer:
    def __init__(self, model: RegressionModel, device: str, log: Logger, batch_size: int, epochs: int, 
                 lr: float, weight_decay: float, step_size: int, gamma: float, accumulation_steps: int) -> None:
        self._model = model
        self._device = device
        self._log = log
        self._batch_size = batch_size
        self._epochs = epochs

        self._optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self._scheduler = lr_scheduler.StepLR(self._optimizer, step_size=step_size, gamma=gamma)

        self._criterion = torch.nn.MSELoss()

        self._lr = lr
        self._weight_decay = weight_decay
        self._accumulation_steps = accumulation_steps

        self._best_valid_loss = float("inf")
        self._best_model = None


    def train(
            self,
            train_dataset: CustomDataset,
            valid_dataset: Optional[CustomDataset] = None
            ) -> None:
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        if valid_dataset is not None:
            valid_loader = DataLoader(valid_dataset, batch_size=self._batch_size, shuffle=False)
        
        self._model.to(self._device)
        for epoch in range(self._epochs):
            self._train_epoch(train_loader, epoch + 1)
            
            if valid_dataset is not None:
                avg_loss = self._valid_epoch(valid_loader, epoch + 1)

                if avg_loss < self._best_valid_loss:
                    self._best_valid_loss = avg_loss
                    self._best_model = {
                        k: v.detach().cpu().clone()
                        for k, v in self._model.state_dict().items()
                    }
                    self._log.write(f"New best model on epoch {epoch + 1}! Validation Loss: {avg_loss:.4f}\n")
            elif epoch == self._epochs - 1:
                self._best_model = {
                        k: v.detach().cpu().clone()
                        for k, v in self._model.state_dict().items()
                    }
                
            self._scheduler.step()

        
    def _train_epoch(self, loader: DataLoader, epoch_idx: int) -> None:
        self._model.train()
        self._log.write(f"Epoch {epoch_idx}\n")
        self._optimizer.zero_grad()
        
        for batch_idx, (feature, label, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col) in enumerate(loader):
            
            feature = feature.to(torch.float).to(self._device)
            padding_mask = padding_mask.to(torch.float).to(self._device)
            n_join_col = n_join_col.to(torch.float).to(self._device).view(-1, 1)
            n_fanout = n_fanout.to(torch.float).to(self._device).view(-1, 1)
            n_table = n_table.to(torch.float).to(self._device).view(-1, 1)
            n_filter_col = n_filter_col.to(torch.float).to(self._device).view(-1, 1)
            pg_est_card = pg_est_card.to(torch.float).to(self._device).view(-1, 1)
            pg_est_card = torch.log(pg_est_card + 1) + 1
            label = torch.log(label.to(torch.float).to(self._device) + 1) + 1
            label = label.view(1, -1)
            
            output = self._model(feature, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col).view(1, -1)
            loss = self._criterion(output, label)
        
            loss = loss / self._accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self._accumulation_steps == 0 or batch_idx == len(loader) - 1:
                self._optimizer.step()
                self._optimizer.zero_grad()
                self._log.write(f"Batch {batch_idx + 1}: loss = {loss.item()}\n")
        

    def _valid_epoch(self, loader: DataLoader, epoch_idx: int) -> None:
        self._model.eval()
        self._log.write(f"Validating epoch {epoch_idx}\n")
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            for batch_idx, (feature, label, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col) in enumerate(loader):
                
                feature = feature.to(torch.float).to(self._device)
                padding_mask = padding_mask.to(torch.float).to(self._device)
                n_join_col = n_join_col.to(torch.float).to(self._device).view(-1, 1)
                n_fanout = n_fanout.to(torch.float).to(self._device).view(-1, 1)
                n_table = n_table.to(torch.float).to(self._device).view(-1, 1)
                n_filter_col = n_filter_col.to(torch.float).to(self._device).view(-1, 1)
                pg_est_card = pg_est_card.to(torch.float).to(self._device).view(-1, 1)
                pg_est_card = torch.log(pg_est_card + 1) + 1
                label = torch.log(label.to(torch.float).to(self._device) + 1) + 1
                label = label.view(1, -1)

                output = self._model(feature, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col).view(1, -1)
                loss = self._criterion(output, label)
                
                self._log.write(f"Validation Batch {batch_idx}: loss = {loss.item()}\n")
                total_loss += float(loss.item()) * label.shape[1]
                total_samples += label.shape[1]
        
        avg_loss = total_loss / total_samples

        self._log.write(f"Average loss: {avg_loss}\n")

        return avg_loss
