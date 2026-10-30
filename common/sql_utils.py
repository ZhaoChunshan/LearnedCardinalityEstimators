import re
from typing import Dict, List

def get_tables_from_comment(sql: str):
    get_tables = lambda sql: list(
        map(
            lambda x: x.strip().split(' ')[1], 
            re.search(r"\/\*(.*)\*\/", sql).group(1).replace("(","").replace(")","").strip().split(",")
        )
    )
    return get_tables(sql)


def single_table_estimation(alias_dict: Dict[str, str], subquery: List[str], est_card: List[float]) -> Dict[str, float]:
    table_estimation = {}
    
    get_tables = lambda sql: list(
            map( lambda x: x.strip().split(' ')[1], 
            re.search(r"\/\*(.*)\*\/", sql).group(1).replace("(","").replace(")","").strip().split(","))
    )

    for sql, card in zip(subquery, est_card):
        tables = get_tables(sql)
        if len(tables) == 1:
            table_estimation[alias_dict[tables[0].strip()]] = card
    
    return table_estimation


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root, y_root = self.find(x), self.find(y)
        if x_root == y_root:
            return
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1
    
    def check(self, x, y):
        return self.find(x) == self.find(y)
    
    def equivalent_groups(self):
        n = len(self.parent)
        groups = [[] for _ in range(n)]
        for i in range(n):
            root_i = self.find(i)
            groups[root_i].append(i)
        return [g for g in groups if len(g) > 0]
