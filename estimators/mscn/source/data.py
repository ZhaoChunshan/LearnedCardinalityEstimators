import torch
import torch.nn.functional as F


from typing import List, Tuple
from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema
from schema.comparison_operator import OPERATOR_ID
from common.torch_utils import create_one_hot
from common.card_utils import normalize_value


class MSCNDataEncoder:
    @staticmethod
    def encode(schema: SQLSchema, queries: List[SQLQuery], sample_number: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        samples_all, predicates_all, joins_all = [], [], []
        samples_masks, predicates_masks, joins_masks = [], [], []
        table_count, column_count, foreign_key_count = schema.table_num(), schema.column_num(), schema.foreign_key_num()
        
        max_sample_num = max(1, max(len(query.nodes()) for query in queries))
        max_predicate_num = max(1, max(sum(len(node.predicates()) for node in query.nodes()) for query in queries))
        max_join_num = max(1, max(len(query.edges()) for query in queries))

        for query in queries:
            tables = torch.vstack([create_one_hot(table_count, node.table().id()) for node in query.nodes()])
            sample_bitmaps = torch.FloatTensor([node.table().samples().evaluate(node.predicates(), sample_number) for node in query.nodes()])
            samples = torch.hstack([sample_bitmaps, tables])

            predicate_tensors_list = [
                torch.hstack(
                    [
                        create_one_hot(column_count, predicate.column.id()),
                        create_one_hot(len(OPERATOR_ID), OPERATOR_ID[predicate.operator]),
                        torch.Tensor([normalize_value(float(predicate.value), predicate.column.min(), predicate.column.max())] \
                                     if not isinstance(predicate.value, str) \
                                     else [predicate.column.value_2_id(predicate.value)/predicate.column.length_of_value_2_id()])
                    ]
                )
                for node in query.nodes() for predicate in node.predicates()]
            if len(predicate_tensors_list) > 0:
                predicates = torch.vstack(predicate_tensors_list)

            join_tensors_list = [create_one_hot(foreign_key_count, edge.foreign_key().id) for edge in query.edges()]
            if len(join_tensors_list) > 0:
                joins = torch.vstack(join_tensors_list)

            # zero padding to max_sample_num, max_predicate_num, max_join_num, and get the mask tensor
            
            samples_padded = F.pad(samples, (0, 0, 0, max_sample_num - samples.shape[0]))
            samples_mask = torch.zeros((samples_padded.shape[0], 1))
            samples_mask[:samples.shape[0], :] = 1

            if len(predicate_tensors_list) > 0:
                predicates_padded = F.pad(predicates, (0, 0, 0, max_predicate_num - predicates.shape[0]))
                predicates_mask = torch.zeros((predicates_padded.shape[0], 1))
                predicates_mask[:predicates.shape[0], :] = 1
            else:
                predicates_padded = torch.zeros((max_predicate_num, column_count + len(OPERATOR_ID) + 1))
                predicates_mask = torch.zeros((predicates_padded.shape[0], 1))

            if len(join_tensors_list) > 0:
                joins_padded = F.pad(joins, (0, 0, 0, max_join_num - joins.shape[0]))
                joins_mask = torch.zeros((joins_padded.shape[0], 1))
                joins_mask[:joins.shape[0], :] = 1            
            else:
                joins_padded = torch.zeros((max_join_num, foreign_key_count))
                joins_mask = torch.zeros((joins_padded.shape[0], 1))
            
            samples_all.append(samples_padded)
            predicates_all.append(predicates_padded)
            joins_all.append(joins_padded)

            samples_masks.append(samples_mask)
            predicates_masks.append(predicates_mask)
            joins_masks.append(joins_mask)

        return torch.stack(samples_all), torch.stack(predicates_all), torch.stack(joins_all), torch.stack(samples_masks), torch.stack(predicates_masks), torch.stack(joins_masks)
