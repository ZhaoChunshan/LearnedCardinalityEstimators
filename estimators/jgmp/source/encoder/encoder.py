import math
import torch


from typing import Tuple, List, Dict
from torch import LongTensor, FloatTensor

from schema.sql_schema import SQLSchema, SQLTable
from query.sql_query import SQLQuery
from common.torch_utils import create_one_hot


class JGMPEncoder:
    def __init__(self, schema: SQLSchema, bitmap_size: int) -> None:
        self._schema = schema
        self._bitmap_size = bitmap_size

    def bitmap_size(self) -> int:
        return self._bitmap_size

    def node_label_encoding_size(self) -> int:
        return len(self._schema.tables())
    
    def edge_label_encoding_size(self) -> int:
        return len(self._schema.foreign_keys())

    def bulk_light_encode_base(self,
                               queries: List[SQLQuery]
                               ) -> Tuple[LongTensor,   # node_index
                                          LongTensor,   # edge_index
                                          FloatTensor,  # edge_labels
                                          LongTensor,   # shared_node_labels
                                          FloatTensor,  # shared_node_cardinalities
                                          FloatTensor,  # shared_node_samples
                                          LongTensor    # my_batch
                                          ]:
        """
            Encode a list of subqueries of some given query.
        """
        # Different from JGMP, we assume one table only occurs once in the join-graph.
        # But we should notice that, one table in different subqueries can be different: virtual occurrence or real occurrence.
        node_index = []
        edge_index = []
        edge_labels = []
        shared_node_labels = []
        shared_node_cardinalities = []
        shared_node_samples = []
        my_batch = []

        node_count, shared_node_count = 0, 0
        shared_node_dict: Dict[Tuple[SQLTable, bool], int] = {}


        for query_count, query in enumerate(queries):
            
            query_table_id = {}
            
            for node in query.nodes():
                table = node.table()
                virtual = node.is_virtual()
                if (table, virtual) not in shared_node_dict:
                    shared_node_dict[(table, virtual)] = shared_node_count
                    shared_node_count += 1
                    shared_node_labels.append(table.id())
                    shared_node_cardinalities.append(math.log(max(node.pg_estimated_card(), 1e-6)))
                    shared_node_samples.append(table.samples().evaluate(node.predicates(), self.bitmap_size()))

                shared_node_id = shared_node_dict[(table, virtual)] 
                node_index.append(shared_node_id)
                my_batch.append(query_count)
                
                query_table_id[table] = node_count
                node_count += 1
            
            for edge in query.edges():
                fk = edge.foreign_key()
                from_tb, to_tb = fk.table, fk.referenced_table
                edge_index.append((query_table_id[from_tb], query_table_id[to_tb]))
                edge_labels.append(create_one_hot(self.edge_label_encoding_size(), fk.id))

        node_index = torch.LongTensor(node_index)

        if edge_index == []:
            edge_index = torch.empty((2, 0), dtype=torch.int64)
            edge_labels = torch.empty((0, self.edge_label_encoding_size()))
        else:
            edge_index = torch.transpose(torch.LongTensor(edge_index), 0, 1)
            edge_labels = torch.stack(edge_labels)

        shared_node_labels = torch.LongTensor(shared_node_labels)
        shared_node_cardinalities = torch.FloatTensor(shared_node_cardinalities)
        shared_node_samples = torch.FloatTensor(shared_node_samples)
        my_batch = torch.LongTensor(my_batch)

        return node_index, edge_index, edge_labels, shared_node_labels, shared_node_cardinalities, shared_node_samples, my_batch
