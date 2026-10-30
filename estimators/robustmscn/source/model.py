import torch
import torch.nn as nn
import torch.nn.functional as F


# Define model architecture, modified for Robust-MSCN

class SetConv(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units,
                 sample_data_features, predicate_column_features,
                 join_bitmap_features, query_masking_ratio: float = 0.0):
        super(SetConv, self).__init__()
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 3 + 1, hid_units)
        self.out_mlp2 = nn.Linear(hid_units + 1, 1)
        self.sample_data_features = sample_data_features
        self.predicate_column_features = predicate_column_features
        self.join_bitmap_features = join_bitmap_features
        self.query_masking_ratio = query_masking_ratio
        self.do_query_masking = False
        
        # For training on dynamic workloads via bitmap shuffling
        self.do_bitmap_shuffling = False

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask, normalized_postgres_est):
        # samples has shape [batch_size x num_tables x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_join_key_groups x join_feats]
        
        if self.do_bitmap_shuffling:
            join_query_feature_num = joins.shape[2] - self.join_bitmap_features
            target_slice = joins[:, :, join_query_feature_num:]
            noise = torch.rand_like(target_slice)
            perm_indices = torch.argsort(noise, dim=-1)
            shuffled_slice = torch.gather(target_slice, dim=-1, index=perm_indices)
            joins = torch.cat(
                (joins[:, :, :join_query_feature_num], shuffled_slice), dim=-1
            )
            
        if self.do_query_masking and self.query_masking_ratio > 0.0:
            samples_mask_flag = (torch.rand(samples.shape[0], samples.shape[1], device=samples.device) > self.query_masking_ratio).float().unsqueeze(-1)
            samples_query_mask = torch.ones_like(samples)
            sample_query_feature_num = samples.shape[2] - self.sample_data_features
            samples_query_mask[:, :, self.sample_data_features:] = (
                samples_mask_flag.expand(-1, -1, sample_query_feature_num)
            )
            samples = samples * samples_query_mask
            
            predicates_mask_flag = (
                torch.rand(predicates.shape[0], predicates.shape[1], device=predicates.device)
                > self.query_masking_ratio
            ).float().unsqueeze(-1)
            predicates_query_mask = torch.ones_like(predicates)
            predicates_query_mask[:, :, :self.predicate_column_features] = (
                predicates_mask_flag.expand(
                    -1, -1, self.predicate_column_features
                )
            )
            predicates = predicates * predicates_query_mask
            
            joins_mask_flag = (
                torch.rand(joins.shape[0], joins.shape[1], device=joins.device)
                > self.query_masking_ratio
            ).float().unsqueeze(-1)
            joins_query_mask = torch.ones_like(joins)
            join_query_feature_num = joins.shape[2] - self.join_bitmap_features
            joins_query_mask[:, :, :join_query_feature_num] = (
                joins_mask_flag.expand(-1, -1, join_query_feature_num)
            )
            joins = joins * joins_query_mask

        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask  # Mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        sample_norm = sample_norm.clamp(min=1)
        hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts
        all_zero_mask = (sample_mask.sum(1) == 0).squeeze(1)
        hid_sample[all_zero_mask] = 0.0

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        predicate_norm = predicate_norm.clamp(min=1)
        hid_predicate = hid_predicate / predicate_norm
        all_zero_mask = (predicate_mask.sum(1) == 0).squeeze(1)
        hid_predicate[all_zero_mask] = 0.0

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        join_norm = join_mask.sum(1, keepdim=False)
        join_norm = join_norm.clamp(min=1)
        hid_join = hid_join / join_norm
        all_zero_mask = (join_mask.sum(1) == 0).squeeze(1)
        hid_join[all_zero_mask] = 0.0

        data_feature = normalized_postgres_est.view(-1, 1)
        hid = torch.cat((hid_sample, hid_predicate, hid_join, data_feature), 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(torch.cat((hid, data_feature), 1)))
        return out

    def enable_query_masking(self):
        self.do_query_masking = True

    def disable_query_masking(self):
        self.do_query_masking = False

    def enable_bitmap_shuffling(self):
        self.do_bitmap_shuffling = True

    def disable_bitmap_shuffling(self):
        self.do_bitmap_shuffling = False
