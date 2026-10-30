import torch
import torch.nn as nn
import torch.nn.functional as F

# Define model architecture


class EnsembleHead(nn.Module):
    """A single ensemble member head with its own fusion layer,
    predicting mean and log-variance."""
    def __init__(self, hid_units: int):
        super(EnsembleHead, self).__init__()
        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.fc_mu = nn.Linear(hid_units, 1)
        self.fc_logvar = nn.Linear(hid_units, 1)

    def forward(self, x: torch.Tensor):
        # x: [batch_size, hid_units * 3]
        hid = F.relu(self.out_mlp1(x))
        mu = torch.sigmoid(self.fc_mu(hid))
        logvar = torch.clamp(self.fc_logvar(hid), min=-10, max=10)
        return mu, logvar


class SetConvEnsemble(nn.Module):
    def __init__(self, sample_feats: int, predicate_feats: int, join_feats: int,
                 hid_units: int, ensemble_size: int):
        super(SetConvEnsemble, self).__init__()

        # Shared backbone: per-modality encoding only (no fusion)
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)

        # K independent ensemble heads
        self.heads = nn.ModuleList([
            EnsembleHead(hid_units) for _ in range(ensemble_size)
        ])

        self._ensemble_size = ensemble_size

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]

        # --- Shared backbone: sample branch ---
        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        sample_norm = sample_norm.clamp(min=1)
        hid_sample = hid_sample / sample_norm
        all_zero_mask = (sample_mask.sum(1) == 0).squeeze(1)
        hid_sample[all_zero_mask] = 0.0

        # --- Shared backbone: predicate branch ---
        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        predicate_norm = predicate_norm.clamp(min=1)
        hid_predicate = hid_predicate / predicate_norm
        all_zero_mask = (predicate_mask.sum(1) == 0).squeeze(1)
        hid_predicate[all_zero_mask] = 0.0

        # --- Shared backbone: join branch ---
        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        join_norm = join_mask.sum(1, keepdim=False)
        join_norm = join_norm.clamp(min=1)
        hid_join = hid_join / join_norm
        all_zero_mask = (join_mask.sum(1) == 0).squeeze(1)
        hid_join[all_zero_mask] = 0.0

        # --- Shared backbone: concat (no fusion — each head does its own) ---
        concat_hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)

        # --- K ensemble heads, each with its own fusion layer ---
        mu_list = []
        logvar_list = []
        for head in self.heads:
            mu, logvar = head(concat_hid)
            mu_list.append(mu)
            logvar_list.append(logvar)

        # [batch_size, ensemble_size] for both
        mu_all = torch.cat(mu_list, dim=1)
        logvar_all = torch.cat(logvar_list, dim=1)

        return mu_all, logvar_all
