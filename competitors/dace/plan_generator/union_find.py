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
