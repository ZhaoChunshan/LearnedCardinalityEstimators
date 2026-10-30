import math
import torch
import torch.nn.functional as F


from typing import List, Tuple
from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema
from schema.comparison_operator import OPERATOR_ID
from synopsis.join_bitmap import JoinBitmap
from common.torch_utils import create_one_hot
from common.card_utils import normalize_value


class RobustMSCNDataEncoder:
    @staticmethod
    def join_key_mapping(schema: SQLSchema):
        join_keys = sorted({
            (fk.referenced_table.name(), fk.referenced_column.name())
            for fk in schema.foreign_keys()
        })
        return {join_key: i for i, join_key in enumerate(join_keys)}

    @staticmethod
    def join_feature_num(schema: SQLSchema, sample_number: int) -> int:
        # Number-of-tables one-hot + involved-tables multi-hot
        # + referenced join-key one-hot + bitmap.
        return sample_number + 2 * schema.table_num() + len(
            RobustMSCNDataEncoder.join_key_mapping(schema)
        )

    @staticmethod
    def encode(schema: SQLSchema, queries: List[SQLQuery], join_bitmap: JoinBitmap) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = [
            RobustMSCNDataEncoder.encode_no_padding(schema, query, join_bitmap)
            for query in queries
        ]
        samples, predicates, joins = zip(*features)
        return RobustMSCNDataEncoder.feature_padding(
            list(samples), list(predicates), list(joins)
        )


    @staticmethod
    def encode_no_padding(schema: SQLSchema, query: SQLQuery, join_bitmap: JoinBitmap) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        table_count, column_count = schema.table_num(), schema.column_num()
        
        tables = torch.vstack([create_one_hot(table_count, node.table().id()) for node in query.nodes()])
        selectivity = torch.FloatTensor([[math.log(max(node.pg_estimated_card() / node.table().cardinality(), 1e-10))] for node in query.nodes()])
        samples = torch.hstack([selectivity, tables])

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
        predicates = torch.vstack(predicate_tensors_list) if len(predicate_tensors_list) > 0 else torch.zeros((0, column_count + len(OPERATOR_ID) + 1))

        join_key_mapping = RobustMSCNDataEncoder.join_key_mapping(schema)
        join_feature_num = RobustMSCNDataEncoder.join_feature_num(
            schema, join_bitmap.sample_number
        )
        join_tensors_list = []

        for join_key, involved_tables, bitmap in join_bitmap.evaluate(query):
            table_count_encoding = create_one_hot(
                table_count, min(len(involved_tables), table_count) - 1
            )
            involved_table_encoding = torch.zeros(table_count)
            for table_name in involved_tables:
                involved_table_encoding[schema.table(table_name).id()] = 1.0

            join_key_encoding = create_one_hot(
                len(join_key_mapping), join_key_mapping[join_key]
            )
            join_tensors_list.append(torch.hstack([
                table_count_encoding,
                involved_table_encoding,
                join_key_encoding,
                torch.FloatTensor(bitmap),
            ]))

        joins = torch.vstack(join_tensors_list) if len(join_tensors_list) > 0 \
            else torch.zeros((0, join_feature_num))

        return samples, predicates, joins


    
    @staticmethod
    def feature_padding(samples_list: List[torch.Tensor], predicates_list: List[torch.Tensor], joins_list: List[torch.Tensor])-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_sample_num = max(1, max(samples.shape[0] for samples in samples_list))
        max_predicate_num = max(1, max(predicates.shape[0] for predicates in predicates_list))
        max_join_num = max(1, max(joins.shape[0] for joins in joins_list))
        samples_all, predicates_all, joins_all = [], [], []
        samples_masks, predicates_masks, joins_masks = [], [], []
        for samples, predicates, joins in zip(samples_list, predicates_list, joins_list):
            samples_padded = F.pad(samples, (0, 0, 0, max_sample_num - samples.shape[0]))
            samples_mask = torch.zeros((samples_padded.shape[0], 1))
            samples_mask[:samples.shape[0], :] = 1

            if predicates.shape[0] > 0:
                predicates_padded = F.pad(predicates, (0, 0, 0, max_predicate_num - predicates.shape[0]))
                predicates_mask = torch.zeros((predicates_padded.shape[0], 1))
                predicates_mask[:predicates.shape[0], :] = 1
            else:
                predicates_padded = torch.zeros((max_predicate_num, predicates.shape[1]))
                predicates_mask = torch.zeros((predicates_padded.shape[0], 1))

            if joins.shape[0] > 0:
                joins_padded = F.pad(joins, (0, 0, 0, max_join_num - joins.shape[0]))
                joins_mask = torch.zeros((joins_padded.shape[0], 1))
                joins_mask[:joins.shape[0], :] = 1            
            else:
                joins_padded = torch.zeros((max_join_num, joins.shape[1]))
                joins_mask = torch.zeros((joins_padded.shape[0], 1))
            
            samples_all.append(samples_padded)
            predicates_all.append(predicates_padded)
            joins_all.append(joins_padded)

            samples_masks.append(samples_mask)
            predicates_masks.append(predicates_mask)
            joins_masks.append(joins_mask)
        samples, predicates, joins, samples_masks, predicates_masks, joins_masks = torch.stack(samples_all), torch.stack(predicates_all), torch.stack(joins_all), torch.stack(samples_masks), torch.stack(predicates_masks), torch.stack(joins_masks)
        return samples, predicates, joins, samples_masks, predicates_masks, joins_masks
