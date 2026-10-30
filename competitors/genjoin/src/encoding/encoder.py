# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: workload-specific positional-encoding normalization.

from abc import ABC
import random
import numpy as np
import networkx as nx
import torch
import pandas as pd

from src.encoding.join_graph import JoinGraph, JoinType
from src.utils.debug import timeit_decorator


def positional_encoding(position, norm: int = 100, const: float = 1e-4):
    """
    The earlier join is made, i.e. the smaller is position, the higher is the positional encoding value
    
    Args:
        norm: making the value of final positional encoding not to be less than 0 (e.g. for JOB with maximum number of joins = 13, value of 100 works well; for STACK value of 50 is better)
    """
    return np.arctan(norm/(position + const)) + 1 - np.arctan(1/const)


class BaseEncoder(ABC):
    @staticmethod
    def encode(join_graph: JoinGraph) -> np.array:
        raise NotImplementedError('Should be implemented on subclass of BaseEncoder.')


class PlanEncoder(BaseEncoder):
    """
    Having the Plan (as of now hints), can encode it into vector corresponding to join types
    """
    # @timeit_decorator
    def encode(join_graph: JoinGraph, join_order: list = None) -> np.array:
        if join_graph.workload_name == 'join_order_benchmark':
            norm = 100
        elif join_graph.workload_name == 'stack':
            norm = 50
        elif join_graph.workload_name == 'job_light':
            norm = 23
        elif join_graph.workload_name == 'stats_ceb':
            norm = 24
        elif join_graph.workload_name == 'ergastf1':
            norm = 28
        else:
            norm = 100
        enc = np.zeros((join_graph.num_undirected_edges, len(JoinType)))
        available_edges = join_graph.get_available_joins(remove_duplicates=False)
        available_joins = {state_edge_idx: [0, 1, 2] for state_edge_idx in available_edges.state_edge_idx.values}
        ranks = np.arange(1, len(JoinType) * len(available_joins) + 1)
        if join_order:
            for join_condition, join_type in join_order:
                # TODO: issue happens with multiple alias joins
                try:
                    left_condition, right_condition = join_condition
                    left_condition = left_condition.strip()
                    right_condition = right_condition.strip()
                    state_edge_idx = available_edges[available_edges.join_condition == (left_condition, right_condition)].state_edge_idx.values[0]
                    # In case of multiple aliases, fail happens here and we do not `use` unnecessarily an additional rank
                    available_joins[state_edge_idx].remove(JoinType[join_type].value)
                    rank, ranks = ranks[0], ranks[1:]
                    enc[state_edge_idx, JoinType[join_type].value] = positional_encoding(rank, norm=norm)
                except:
                    continue
        else:
            # First assign higher ranks to 1 out of 3 join possibilities for each unordered pair
            for state_edge_idx in available_edges.state_edge_idx.unique():
                # TODO: issue happens with multiple alias joins, same as above
                try:
                    # action_edge_idx = random.choice(available_edges[available_edges.state_edge_idx == state_edge_idx].action_edge_idx.values)
                    i = random.choice([0, 1, 2])
                    available_joins[state_edge_idx].remove(i)
                    rank, ranks = ranks[0], ranks[1:]
                    enc[state_edge_idx, i] = positional_encoding(rank, norm=norm)
                except:
                    continue
        for state_edge_idx, join_types in available_joins.items():
            for i in join_types:
                rank_index = np.random.randint(0, len(ranks))
                rank = ranks[rank_index]
                ranks = np.delete(ranks, rank_index)
                enc[state_edge_idx, i] = positional_encoding(rank, norm=norm)

        out = enc.reshape(np.multiply(*enc.shape))
        out = torch.from_numpy(out)\
                   .to(dtype=torch.float32)
                    
        return out


class QueryEncoder(BaseEncoder):
    """
    Encodes the conditional vector as a query information, i.e. without actual joins performed.
    
    Should be performed on a JoinGraph where no joins yet performed, as it uses all edges
    that are available for joining.
    """
    # @timeit_decorator
    @staticmethod
    def encode(join_graph: JoinGraph) -> np.array:
        enc = np.zeros((join_graph.num_undirected_edges, 3))
        
        available_edges = join_graph.get_available_joins(remove_duplicates=True)
        if available_edges.shape[0] > 0:
            # available_edge_idx = np.sort(available_edges['state_edge_idx'].unique())
            available_edges = available_edges[available_edges['join_directionality'] == 0].sort_values(['state_edge_idx'])
            enc[available_edges['state_edge_idx'], 0] = 1

            # Encoding Positions 1&2: Selectivities of the Join tables, including
            # their filters applied (estimates from DB)
            #
            enc[available_edges['state_edge_idx'], 1] = available_edges['left_selectivity']
            enc[available_edges['state_edge_idx'], 2] = available_edges['right_selectivity']
        out = enc.reshape(np.multiply(*enc.shape))
        out = torch.from_numpy(out)\
                   .to(dtype=torch.float32)
        
        return out


class PSTEncoder(BaseEncoder):
    """
    Possibility, Selectivity, Type => P.S.T.

    Takes a JoinGraph and encodes all its edges based on
    the possibility if the join could be part of it, the
    selectivity of the concerned tables and the type of join.
    """
    
    @staticmethod
    def encode(join_graph: JoinGraph) -> np.array:
        enc = np.zeros((join_graph.num_undirected_edges, 7))

        # Duplicate handling: Which of the query join edges do we encode in case
        # that there exist duplicate edges (multiple aliases for the same tables 
        # joining on the same relation edge)
        #
        # --> HERE: Add Thompson sampling idea, as we currently remove
        # duplicates on the same edge (until they are joined and allow for the
        # same edge to be encoded again)
        #
        available_edges = join_graph.get_available_joins(remove_duplicates=True)

        # Encoding Position 0: Is the Join possible?
        # -1: Not Possible
        #  0: Not yet Joined
        #  1: Already Joined
        #
        # Not Possible Joins, initially set all to impossible
        enc[:, 0] = -1

        # Already Joined
        if len(join_graph.join_history) > 0:
            # performed_join_idx = np.unique(np.sort([h.state_edge_idx for h in join_graph.join_history]))
            # enc[performed_join_idx, 0] = 1

            # Adding the selectivities for already performed join edges
            # from the join history
            init_edges = join_graph.initial_join_edges
            for position, h in enumerate(join_graph.join_history):
                # 2024-01-05 suli: adding simple positional encoding, to be passed in state instead of keeping last N joins as steps
                enc[h.state_edge_idx, 0] = positional_encoding(position)
                rows = init_edges[init_edges['join_condition'] == h.join_condition]
                if len(rows) == 1:
                    row = rows.iloc[0]
                    if h.join_directionality == 0:
                        enc[h.state_edge_idx, 1] = row.left_selectivity
                        enc[h.state_edge_idx, 2] = row.right_selectivity
                    else:
                        enc[h.state_edge_idx, 1] = row.right_selectivity
                        enc[h.state_edge_idx, 2] = row.left_selectivity


        # Not Yet Joined (overwrites potentially performed joins)
        # --> after already joined edges are written
        #
        if available_edges.shape[0] > 0:
            # available_edge_idx = np.sort(available_edges['state_edge_idx'].unique())
            available_edges = available_edges[available_edges['join_directionality'] == 0].sort_values(['state_edge_idx'])
            enc[available_edges['state_edge_idx'], 0] = 0

            # Encoding Positions 1&2: Selectivities of the Join tables, including
            # their filters applied (estimates from DB)
            #
            enc[available_edges['state_edge_idx'], 1] = available_edges['left_selectivity']
            enc[available_edges['state_edge_idx'], 2] = available_edges['right_selectivity']
        
        # Include the join directionality
        # Using lexigraphical order of join tables/cols:
        # 0 -> (left, right)
        # 1 -> (right, left)
        # 
        # Set all to zero, then add the directionality according to the join history
        # lookup against the db_graph edgelist
        enc[:, 3] = 0
        
        if len(join_graph.join_history) > 0:
            for h in join_graph.join_history:
                enc[h.state_edge_idx, 3] = h.join_directionality

        # Set the One-Hot encoded Join Type at positions 4-6
        if len(join_graph.join_history) > 0:
            for h in join_graph.join_history:
                enc[h.state_edge_idx, 4 + h.join_type.value] = 1

        out = enc.reshape(np.multiply(*enc.shape))
        out = torch.from_numpy(out)\
                   .to(dtype=torch.float32)
        
        return out
    

class PSTHistogramEncoder(BaseEncoder):
    """
    Possibility, Selectivity, Type => P.S.T.

    Takes a JoinGraph and encodes all its edges based on
    the possibility if the join could be part of it, the
    selectivity of the concerned tables and the type of join.
    """
    
    @staticmethod
    def encode(join_graph: JoinGraph) -> np.array:
        # Load the database graph and separate the join condition into column names 
        # Ensure, that the DF is sorted by the :action_edge_idx
        #
        df_db = nx.to_pandas_edgelist(join_graph.database_graph)
        df_db = df_db.sort_values(['action_edge_idx']).reset_index(drop=True)

        # Calculate total size of encoding from scott histogram sizes
        encoding_size = (df_db.shape[0] * 4) + df_db['left_hist_size'].sum() + df_db['right_hist_size'].sum()
        offsets = df_db['offset'].tolist()

        enc = np.zeros(encoding_size)

        available_edges = join_graph.get_available_joins(remove_duplicates=True)
        for i, row in df_db.iterrows():
            block_offset = offsets[i]
            next_block_offset = offsets[i+1] if i < df_db.shape[0]-1 else 0
            left_hist_offset = block_offset + 1
            right_hist_offset = block_offset + 1 + row['left_hist_size']

            # Encoding Position 0: Is the Join possible?
            # =========================================================================================================
            # -1: Not Possible
            #  0: Not yet Joined
            #  1: Already Joined
            #
            # Default case: mark anything as not possible and overwrite below
            enc[block_offset] = -1

            # Check that there are still available edges
            if (not available_edges.empty) and (row['action_edge_idx'] in available_edges['action_edge_idx'].unique()):
                # Join not yet performed
                enc[block_offset] = 0
            else:
                if len(join_graph.join_history) > 0:
                    for h in join_graph.join_history:
                        h_join = join_graph.initial_join_edges[join_graph.initial_join_edges['join_condition'] == h.join_condition].iloc[0]
                        if row['join_condition'] == h_join['join_condition__resolved']:
                            # Join edge has been used already
                            enc[block_offset] = 1


            # Encoding Left & Right Histograms
            # =========================================================================================================
            #
            # Only the available and already joined edges contain nonzero values in the
            # histograms, others are kept at the default zero
            #
            left_hist = None
            right_hist = None

            # Is the current edge still available to be joined?
            if not available_edges.empty:
                available_edge = available_edges[available_edges['join_condition__resolved'] == row['join_condition']]
                if len(available_edge) == 1:
                    available_edge = available_edge.iloc[0]
                    left_hist = available_edge['left_selectivity']
                    right_hist = available_edge['right_selectivity']

            # Is the current edge already joined?
            if len(join_graph.join_history) > 0:
                for h in join_graph.join_history:
                    h_join = join_graph.initial_join_edges[join_graph.initial_join_edges['join_condition'] == h.join_condition].iloc[0]
                    if row['join_condition'] == h_join['join_condition__resolved']:
                        left_hist = h_join['left_selectivity']
                        right_hist = h_join['right_selectivity']
                        break

            # Encoding Left Histogram
            if left_hist is not None:
                assert enc[block_offset] != -1, f"""
                Histograms should not be encoded if the join is not possible in the first place.
                Is this join not possible? {row['join_condition']}
                """
                assert left_hist.shape[0] == row['left_hist_size'], f"""
                Left side histogram of {row['join_condition']} has a different size than expected.
                    (actual: {left_hist.shape[0]}, expected: {row['left_hist_size']})
                """
                enc[left_hist_offset:left_hist_offset+row['left_hist_size']] = left_hist

            # Encoding Right Histogram
            if right_hist is not None:
                assert enc[block_offset] != -1, f"""
                Histograms should not be encoded if the join is not possible in the first place.
                Is this join not possible? {row['join_condition']}
                """
                assert right_hist.shape[0] == row['right_hist_size'], f"""
                Right side histogram of {row['join_condition']} has a different size than expected.
                    (actual: {right_hist.shape[0]}, expected: {row['right_hist_size']})
                """
                enc[right_hist_offset:right_hist_offset+row['right_hist_size']] = right_hist


            # Encoding (used) Join Types
            # =========================================================================================================
            #
            join_type = None
            if len(join_graph.join_history) > 0:
                for h in join_graph.join_history:
                    h_join = join_graph.initial_join_edges[join_graph.initial_join_edges['join_condition'] == h.join_condition].iloc[0]
                    if row['join_condition'] == h_join['join_condition__resolved']:
                        join_type = h.join_type
                        break

                if join_type is not None:
                    enc[next_block_offset - 3 + join_type.value] = 1

            # print(offsets)
            # print('---'*30)
            # print(row['join_condition'])
            # np.set_printoptions(suppress=True, precision=1)
            # print(enc[block_offset:block_offset+4+row['left_hist_size']+row['right_hist_size']])
            # print()

        out = torch.from_numpy(enc)\
                   .to(dtype=torch.float32)
        
        return out
