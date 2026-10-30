import os
import random
from enum import Enum
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from functools import lru_cache

from src.database.query_loader import QueryLoader
from src.database.selectivity_extractor import PostgresSelectivityExtractor, SelectivityExtractor, PostgresHistogramExtractor
from src.utils.query_helper import parse_sql_query
from src.utils.path_helper import WORKLOADS, DATABASES
from src.encoding.database_graph import DatabaseGraph


from src.utils.debug import timeit_decorator


class JoinCompleteException(Exception):
    "Raised when there are no more joins left to do."
    pass


class RedundantJoinException(Exception):
    """
    Raised when a join is performed that is technically valid but does
    not increase the spanning tree of the join, i.e. when so far the joined
    tables are A-B-C and the join A-C is added.
    """
    pass


class JoinNotPossibleException(Exception):
    "Raised when trying to join via an impossible condition."
    pass


class JoinType(Enum):
    HashJoin = 0
    MergeJoin = 1
    NestLoop = 2


class ScanType(Enum):
    IndexScan = 0
    SeqScan = 1


@dataclass
class PerformedJoin:
    available_join_idx: int
    action_edge_idx: int
    state_edge_idx: int
    join_type: JoinType
    join_condition: tuple
    join_directionality: int
    contracted_node_name: str


class JoinGraph:
    # @timeit_decorator
    def __init__(self, query: str, workload_name: str = 'join_order_benchmark', selectivity_extractor: SelectivityExtractor = None, alias_selectivities: dict = None, use_histograms: bool = False):
        self.query = query
        self.workload_name = workload_name
        self.database_name = DATABASES[self.workload_name]
        self.use_histograms = use_histograms

        self._parse_query(self.query)
        
        if not selectivity_extractor:
            if self.use_histograms:
                selectivity_extractor = PostgresHistogramExtractor(workload_name=self.workload_name, database_name=self.database_name)
            else:
                selectivity_extractor = PostgresSelectivityExtractor(workload_name=self.workload_name, database_name=self.database_name)
        if alias_selectivities is not None:
            self.alias_selectivities = alias_selectivities
        else:
            if self.use_histograms:
                self.alias_selectivities = selectivity_extractor.get_true_selectivity(self.query).copy()
            else:
                self.alias_selectivities = selectivity_extractor.get_estimated_selectivity(self.query, alias_dict=self.alias_dict, filter_conditions=self.filter_conditions).copy()

        self.full_selectivities = selectivity_extractor.full_cardinalities
        self.selectivity_extractor = selectivity_extractor
        # del selectivity_extractor
        self._load_database_graph()
        self.initialize_graph()


    # @timeit_decorator
    def initialize_graph(self):
        self.reset_graph()

        # Look for the join edges in the DB graph that are part of this query
        self.update_available_edges()
        self.initial_join_edges = nx.to_pandas_edgelist(self.query_graph)


    # @timeit_decorator
    def reset_graph(self):
        self.join_history = []
        self.is_done = False

        # The query_graph initially holds all potential join edges and iteratively
        # loses them, as Joins are performed
        #
        self.query_graph = nx.MultiDiGraph()

        # The join_graph initially holds no edges, but slowly builds them up as
        # Joins are performed, eventually forming a spanning tree across the nodes.
        # (only "executed" nodes are added, not transitively removed ones)
        #
        self.join_graph = nx.DiGraph()

        # Add all aliases as nodes in the join graph
        for table_alias in self.alias_dict.keys():
            self.query_graph.add_node(table_alias)
            self.join_graph.add_node(table_alias)

        # Add all potential join edges to the join graph
        for cond in self.join_conditions:
            left = min(cond['columns'])
            right = max(cond['columns'])
            left_alias, left_column = left.split('.')
            right_alias, right_column = right.split('.')

            left_resolved = f"{self.alias_dict[left_alias]}.{left_column}"
            right_resolved = f"{self.alias_dict[right_alias]}.{right_column}"

            if self.use_histograms:
                left_selectivity = self.alias_selectivities[left]
                right_selectivity = self.alias_selectivities[right]
                # In case of histograms, generate the order-by-attribute prod_selectivity by summing the histograms up
                prod_selectivity = sum(left_selectivity) * sum(right_selectivity)

            else:
                left_selectivity = self.alias_selectivities[left_alias]
                right_selectivity = self.alias_selectivities[right_alias]
                prod_selectivity = left_selectivity * right_selectivity

            df_db = nx.to_pandas_edgelist(self.database_graph)
            # We explicitly add both directions to the graph, as the direction
            # can have a profound impact on the execution time
            #
            df_db__edge = df_db[df_db['join_condition'] == (left_resolved, right_resolved)]
            if len(df_db__edge) == 0:
                raise JoinNotPossibleException(f"Could not find join condition {(left_resolved, right_resolved)} in the configured joins.")
            
            # Ensuring that the edge from (left-right) only exists at most once
            if not self.query_graph.has_edge(left_alias, right_alias):
                self.query_graph.add_edge(
                    left_alias,
                    right_alias,
                    left_alias=left_alias,
                    right_alias=right_alias,
                    join_condition=(left, right),
                    join_condition__resolved=(left_resolved, right_resolved),
                    left_selectivity=left_selectivity,
                    right_selectivity=right_selectivity,
                    prod_selectivity=prod_selectivity,
                    state_edge_idx=df_db__edge.iloc[0]['state_edge_idx'],
                    action_edge_idx=df_db__edge.iloc[0]['action_edge_idx'],
                    join_directionality=df_db__edge.iloc[0]['join_directionality']
                )

            # Edge with inverted directionality
            df_db__edge = df_db[df_db['join_condition'] == (right_resolved, left_resolved)]
            if len(df_db__edge) == 0:
                raise JoinNotPossibleException(f"Could not find join condition {(left_resolved, right_resolved)} in the configured joins.")
            
            # Ensuring that the edge from (right-left) only exists at most once
            if not self.query_graph.has_edge(right_alias, left_alias):
                self.query_graph.add_edge(
                    right_alias,
                    left_alias,
                    left_alias=right_alias,
                    right_alias=left_alias,
                    join_condition=(right, left),
                    join_condition__resolved=(right_resolved, left_resolved),
                    left_selectivity=right_selectivity,
                    right_selectivity=left_selectivity,
                    prod_selectivity=prod_selectivity,
                    state_edge_idx=df_db__edge.iloc[0]['state_edge_idx'],
                    action_edge_idx=df_db__edge.iloc[0]['action_edge_idx'],
                    join_directionality=df_db__edge.iloc[0]['join_directionality']
                )



    # @timeit_decorator
    def add_join(self, selected_join: pd.Series, join_type: JoinType = JoinType.HashJoin) -> None:
        """
        Method to add a join to the JoinGraph ('act as if one was performed') and 
        relevant changes to the `:query_graph` and `:join_graph` Graphs.

        The `:selected_join` parameter expects a pd.Series that contains the `source`, `target`
        and `join_condition` attributes at least and is ideally made by filtering the returned
        dataframe from the `#get_available_joins()` method.
        """
        if self.is_done:
            raise JoinCompleteException("There are no more Joins left to do.")

        # Remove executed Join from query graph and add to the join_graph
        # (we specifically opted for contracting the node names to follow the
        # "Leading" syntax of PgHintPlan for ease of use)
        #
        selected_source = selected_join['source'].item()
        selected_target = selected_join['target'].item()

        contracted_node_name = f"({selected_source} {selected_target})"

        # Add Join to history of joins
        self.join_history.append(
            PerformedJoin(
                available_join_idx=selected_join.index.item(),
                action_edge_idx=int(selected_join['action_edge_idx']),
                state_edge_idx=int(selected_join['state_edge_idx']),
                join_type=join_type,
                join_condition=selected_join['join_condition'].item(),
                join_directionality=int(selected_join['join_directionality'].item()),
                contracted_node_name=contracted_node_name
            )
        )

        # Add relevant edge to Join Graph
        join_condition = selected_join['join_condition'].item()
        left_join = min(join_condition)
        right_join = max(join_condition)

        self.join_graph.add_edge(left_join.split('.')[0], right_join.split('.')[0])

        # Remove both connections of chosen join (a,b), taking into
        # account both directions can no longer be used
        #
        self.query_graph.remove_edge(selected_source, selected_target)
        self.query_graph.remove_edge(selected_target, selected_source)
        self.query_graph = nx.contracted_nodes(self.query_graph, selected_source, selected_target, False)
        self.query_graph = nx.relabel_nodes(self.query_graph, {selected_source: contracted_node_name})

        # Helper method for checking transitively connected nodes
        def check_edge(edge):
            source, target = edge
            sources = source.replace('(', '').replace(')', '').split(' ')
            targets = target.replace('(', '').replace(')', '').split(' ')

            for s in sources:
                for t in targets:
                    if nx.node_connectivity(self.join_graph, s, t) != 0:
                        self.query_graph.remove_edge(source, target)
                        print(f"check_edge: Removed transitive edge for {s}->{t} (forward)")
                    if nx.node_connectivity(self.join_graph, t, s) != 0:
                        self.query_graph.remove_edge(target, source)
                        print(f"check_edge: Removed transitive edge for {t}->{s} (reverse)")
                    
        # Also remove edges that are transitively connected now
        edges_to_check = self.query_graph.edges()
        for edge in edges_to_check:
            check_edge(edge)

        if len(self.join_history) >= len(self.alias_dict.keys()) - 1:
            self.is_done = True


    # @timeit_decorator
    def add_join_by_edge_id(self, action_edge_idx: int, join_type: JoinType = JoinType.HashJoin) -> None:
        """
        Adds a join to the `JoinGraph`, identified by the `action_edge_idx` from the edgelist
        returned by `#get_available_joins()`.
        
        Example: `add_join_by_condition(14)` for the join `kt1.id = t1.kind_id`
        """
        if self.is_done:
            raise JoinCompleteException("There are no more Joins left to do.")

        df_available = self.get_available_joins()
        selected_join = df_available[df_available['action_edge_idx'] == action_edge_idx]

        if len(selected_join) != 1:
            raise ValueError(f"The provided action_edge_idx {action_edge_idx} is not a valid edge to be used.")

        self.add_join(selected_join, join_type)


    # @timeit_decorator
    def add_join_by_condition(self, join_tuple: tuple, join_type: JoinType = JoinType.HashJoin) -> None:
        """
        Adds a join to the `JoinGraph`, identified by the join condition. Note that the same form should be
        used as in the data of the edgelist returned by `#get_available_joins()`.
        
        Example: `add_join_by_condition('kt1.id', 't1.kind_id')` for the join `kt1.id = t1.kind_id`
        """
        if self.is_done:
            raise JoinCompleteException("There are no more Joins left to do.")

        df_available = nx.to_pandas_edgelist(self.query_graph)
        selected_join = df_available[df_available['join_condition'] == join_tuple]

        # If there was no selected_join found (or multiple), check if the join
        # was available initially (=RedundantJoin) or not at all (JoinNotPossible)
        #
        if len(selected_join) != 1:
            available_initially = self.initial_join_edges[self.initial_join_edges['join_condition'] == join_tuple]
            if len(available_initially) != 0:
                raise RedundantJoinException(f"The provided join_tuple '{join_tuple}' is redundant, as the two aliases have already been connected via different joins.")

            raise JoinNotPossibleException(f"The provided join_tuple '{join_tuple}' is not a valid edge to be used.")

        self.add_join(selected_join, join_type)


    # @timeit_decorator
    def get_available_joins(self, remove_duplicates=True) -> pd.DataFrame:
        """
        Return the edge data of available Joins, that are filtered down to the ones that
        are being encoded. Note: That means, not all (truly) possible Joins are returned,
        as multiple of the same table combinations could be aliased differently but only
        one of those will be shown, each.
        """
        filtered_query_join_edges = nx.to_pandas_edgelist(self.query_graph)

        if filtered_query_join_edges.shape[0] > 0:
            filtered_query_join_edges['action_edge_idx'] = filtered_query_join_edges['action_edge_idx'].astype(int)

            filtered_query_join_edges = filtered_query_join_edges \
                .sort_values(['action_edge_idx', 'prod_selectivity', 'source', 'target']) \
            
            if remove_duplicates:
                filtered_query_join_edges = filtered_query_join_edges \
                    .drop_duplicates(subset='action_edge_idx', keep='first')

        return filtered_query_join_edges.sort_index()


    # @timeit_decorator
    def update_available_edges(self) -> None:
        """
        Iterates through the edges in the join graph and adds the edge index in the possible
        edges of the DB graph for reference. Note: Multiple of the same edge can be chosen,
        as tables can have different aliases, but contain the same types of join connections.
        """
        available_edge_idx = set()
        df_db = nx.to_pandas_edgelist(self.database_graph)

        query_join_edges = nx.to_pandas_edgelist(self.query_graph)

        for i, row in query_join_edges.iterrows():
            try:
                possible_idx = df_db[df_db['join_condition'] == row['join_condition__resolved']].iloc[0]['action_edge_idx']
            except IndexError:
                raise JoinNotPossibleException(f"""
                    Attempted to initialize a JoinGraph with a JOIN that is not present in the
                    :database_graph defined: {row['join_condition__resolved']}
                """)
            query_join_edges.loc[i, 'action_edge_idx'] = possible_idx
            available_edge_idx.add(possible_idx)

        # Store the initially available edge ids
        self.initial_edge_idx = np.array(sorted(available_edge_idx), dtype=int)

        # Only update, if there are still edges in the graph, otherwise it will have been
        # contracted to a single node graph without edges
        #
        if query_join_edges.shape[0] > 0:
            # Replace join graph with edge list that now also include the edge_idx
            self.query_graph = nx.from_pandas_edgelist(query_join_edges, edge_attr=True, create_using=nx.MultiDiGraph())


    # @timeit_decorator
    def _parse_query(self, query: str) -> None:
        """
        Parse the query using our helpers, such that the relevant information is available, i.e. how
        aliases are translated back to table identifiers, which tables are involved and what join and
        filter conditions there exist.
        """
        # import code; code.interact(local=dict(globals(), **locals()))
        query_tables, alias_dict, query_conditions = parse_sql_query(query)

        self.query_tables = query_tables
        self.alias_dict = alias_dict
        self.query_conditions = query_conditions

        self.join_conditions = [cond for cond in self.query_conditions if cond['is_join']]
        self.filter_conditions = [cond for cond in self.query_conditions if not cond['is_join']]
        

    # @timeit_decorator
    def _load_database_graph(self) -> None:
        """
        Read an existing nx.MultiDiGraph that contains all the possible JOIN edges or generate it, in case it does not exist yet.
        """
        database_graph_path = f"data/processed/{WORKLOADS[self.workload_name]}/possible_joins__directed.edgelist"

        if not os.path.exists(database_graph_path):
            dbg = DatabaseGraph(self.selectivity_extractor, self.workload_name)
            dbg.build_edgelist()
            del dbg

        self.database_graph = nx.read_edgelist(path=database_graph_path, create_using=nx.MultiDiGraph())
        self.num_undirected_edges = nx.to_pandas_edgelist(self.database_graph)['state_edge_idx'].nunique()


    # @timeit_decorator
    def action_from_edge(self, edge_idx, join_type: JoinType = JoinType.HashJoin):
        """
        Converts from an `edge_idx` (and an optional `JoinType`)
        to the `action_idx` as part of the encoding configuration
        """
        join_type_idx = join_type.value
        return len(JoinType) * edge_idx + join_type_idx


    # @timeit_decorator
    def edge_from_action(self, action_idx):
        """
        Takes an `action_idx` as part of the encoding configuration
        and returns the corresponding `edge_idx` and `JoinType`
        """
        edge_idx, join_type_idx = divmod(action_idx, len(JoinType))
        return edge_idx, JoinType(join_type_idx)
            

class SimpleJoinGraph:
    # @timeit_decorator
    def __init__(self, query: str, workload_name: str = 'join_order_benchmark', selectivity_extractor: SelectivityExtractor = None, alias_selectivities: dict = None, database_graph: DatabaseGraph = None):
        self.query = query
        self.workload_name = workload_name
        self.database_name = DATABASES[self.workload_name]
        self._get_available_joins__cache = lru_cache(maxsize=2)(self._get_available_joins)

        self._parse_query(self.query)

        if not selectivity_extractor:
            selectivity_extractor = PostgresSelectivityExtractor(workload_name=self.workload_name, database_name=self.database_name)
        
        if alias_selectivities is not None:
            self.alias_selectivities = alias_selectivities
        else:
            self.alias_selectivities = selectivity_extractor.get_estimated_selectivity(self.query, alias_dict=self.alias_dict, filter_conditions=self.filter_conditions).copy()

        self.full_selectivities = selectivity_extractor.full_cardinalities
        self.selectivity_extractor = selectivity_extractor
        self._load_database_graph(database_graph)
        self.initialize_graph()


    @property
    def num_undirected_edges(self):
        return self.database_graph.num_undirected_edges


    # @timeit_decorator
    def initialize_graph(self):
        self.reset_graph()

        # Look for the join edges in the DB graph that are part of this query
        self.update_available_edges()


    # @timeit_decorator
    def reset_graph(self):
        self.is_done = False

        # The query_graph initially holds all potential join edges and iteratively
        # loses them, as Joins are performed
        #
        self.query_graph = nx.MultiDiGraph()

        # Add all aliases as nodes in the join graph
        for table_alias in self.alias_dict.keys():
            self.query_graph.add_node(table_alias)

        # Add all potential join edges to the join graph
        for cond in self.join_conditions:
            left = min(cond['columns'])
            right = max(cond['columns'])
            left_alias, left_column = left.split('.')
            right_alias, right_column = right.split('.')

            left_resolved = f"{self.alias_dict[left_alias]}.{left_column}"
            right_resolved = f"{self.alias_dict[right_alias]}.{right_column}"

            left_selectivity = self.alias_selectivities[left_alias]
            right_selectivity = self.alias_selectivities[right_alias]

            df_db = self.database_graph.edge_list
            # We explicitly add both directions to the graph, as the direction
            # can have a profound impact on the execution time
            #
            df_db__edge = df_db[df_db['join_condition'] == (left_resolved, right_resolved)]
            if len(df_db__edge) == 0:
                raise JoinNotPossibleException(f"Could not find join condition {(left_resolved, right_resolved)} in the configured joins.")
            
            # Ensuring that the edge from (left-right) only exists at most once
            if not self.query_graph.has_edge(left_alias, right_alias):
                self.query_graph.add_edge(
                    left_alias,
                    right_alias,
                    left_alias=left_alias,
                    right_alias=right_alias,
                    join_condition=(left, right),
                    join_condition__resolved=(left_resolved, right_resolved),
                    left_selectivity=left_selectivity,
                    right_selectivity=right_selectivity,
                    state_edge_idx=df_db__edge.iloc[0]['state_edge_idx'],
                    action_edge_idx=df_db__edge.iloc[0]['action_edge_idx'],
                    join_directionality=df_db__edge.iloc[0]['join_directionality']
                )

            # Edge with inverted directionality
            df_db__edge = df_db[df_db['join_condition'] == (right_resolved, left_resolved)]
            if len(df_db__edge) == 0:
                raise JoinNotPossibleException(f"Could not find join condition {(left_resolved, right_resolved)} in the configured joins.")
            
            # Ensuring that the edge from (right-left) only exists at most once
            if not self.query_graph.has_edge(right_alias, left_alias):
                self.query_graph.add_edge(
                    right_alias,
                    left_alias,
                    left_alias=right_alias,
                    right_alias=left_alias,
                    join_condition=(right, left),
                    join_condition__resolved=(right_resolved, left_resolved),
                    left_selectivity=right_selectivity,
                    right_selectivity=left_selectivity,
                    state_edge_idx=df_db__edge.iloc[0]['state_edge_idx'],
                    action_edge_idx=df_db__edge.iloc[0]['action_edge_idx'],
                    join_directionality=df_db__edge.iloc[0]['join_directionality']
                )

    # @timeit_decorator
    def get_available_joins(self, remove_duplicates=True) -> pd.DataFrame:
        results, results_without_duplicates = self._get_available_joins__cache()
        if remove_duplicates:
            return results_without_duplicates
        return results


    # @timeit_decorator
    def _get_available_joins(self) -> pd.DataFrame:
        """
        Return the edge data of available Joins, that are filtered down to the ones that
        are being encoded. Note: That means, not all (truly) possible Joins are returned,
        as multiple of the same table combinations could be aliased differently but only
        one of those will be shown, each.
        """
        filtered_query_join_edges = nx.to_pandas_edgelist(self.query_graph)

        if filtered_query_join_edges.shape[0] > 0:
            filtered_query_join_edges['action_edge_idx'] = filtered_query_join_edges['action_edge_idx'].astype(int)

            filtered_query_join_edges = filtered_query_join_edges \
                .sort_values(['action_edge_idx', 'source', 'target']) \
            
            wihout_duplicates = filtered_query_join_edges.copy() \
                .drop_duplicates(subset='action_edge_idx', keep='first')

            return filtered_query_join_edges.sort_index(), wihout_duplicates.sort_index()
        
        else:
            return filtered_query_join_edges, filtered_query_join_edges


    # @timeit_decorator
    def update_available_edges(self) -> None:
        """
        Iterates through the edges in the join graph and adds the edge index in the possible
        edges of the DB graph for reference. Note: Multiple of the same edge can be chosen,
        as tables can have different aliases, but contain the same types of join connections.
        """
        available_edge_idx = set()
        df_db = self.database_graph.edge_list

        query_join_edges = nx.to_pandas_edgelist(self.query_graph)

        for i, row in query_join_edges.iterrows():
            try:
                possible_idx = df_db[df_db['join_condition'] == row['join_condition__resolved']].iloc[0]['action_edge_idx']
            except IndexError:
                raise JoinNotPossibleException(f"""
                    Attempted to initialize a JoinGraph with a JOIN that is not present in the
                    :database_graph defined: {row['join_condition__resolved']}
                """)
            query_join_edges.loc[i, 'action_edge_idx'] = possible_idx
            available_edge_idx.add(possible_idx)

        # Only update, if there are still edges in the graph, otherwise it will have been
        # contracted to a single node graph without edges
        #
        if query_join_edges.shape[0] > 0:
            # Replace join graph with edge list that now also include the edge_idx
            self.query_graph = nx.from_pandas_edgelist(query_join_edges, edge_attr=True, create_using=nx.MultiDiGraph())


    # @timeit_decorator
    def _parse_query(self, query: str) -> None:
        """
        Parse the query using our helpers, such that the relevant information is available, i.e. how
        aliases are translated back to table identifiers, which tables are involved and what join and
        filter conditions there exist.
        """
        query_tables, alias_dict, query_conditions = parse_sql_query(query)

        self.query_tables = query_tables
        self.alias_dict = alias_dict
        self.query_conditions = query_conditions

        self.join_conditions = [cond for cond in self.query_conditions if cond['is_join']]
        self.filter_conditions = [cond for cond in self.query_conditions if not cond['is_join']]


    # @timeit_decorator
    def _load_database_graph(self, database_graph: DatabaseGraph = None) -> None:
        if database_graph is None:
            self.database_graph = DatabaseGraph(selectivity_extractor=self.selectivity_extractor, workload_name=self.workload_name)
        else:
            self.database_graph = database_graph


def random_join(schema_graph, n_joins=2):
    """
    Provides an artifitially created set of tables and corresponding
    join conditions based on random walk over database graph, having random 
    start point and not more than `n_joins` joins
    
    The `neighbours` to be chosen from are continuously updated when new node visited,
    subtracting the set of visited nodes themselves
    """
    visited = set()
    join_conditions = list()

    # Randomly finding initial node
    prev_node = random.choice(list(schema_graph.nodes))
    visited.add(prev_node)
    neighbours = set(schema_graph.neighbors(prev_node))

    while n_joins and neighbours:
        next_node = random.sample(neighbours, 1)[0]
        visited.add(next_node)
        n_joins -= 1
        # need to add new neighbours to list, and get rid of visited nodes
        neighbours.update(set(schema_graph.neighbors(next_node)))
        neighbours.difference_update(visited)
        prev_node = next_node
        
    join_conditions = []
    subgraph = nx.subgraph(schema_graph, visited)
    for edge in subgraph.edges():
        # taking only first condition out of multiples if exist
        edge_data = list(dict(subgraph.get_edge_data(*edge)).values())[0]
        join_conditions.append(edge_data['join_condition'])
    join_conditions = list(map(lambda x: ' = '.join(x), join_conditions))
        
    return visited, list(set(join_conditions))
