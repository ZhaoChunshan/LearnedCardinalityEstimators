#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define MAX_RELATION_NUM 64
#define CONSTRUCT_PLAN_SPACE_LIMIT_SIZE 16
#define INVALID_SUBQUERY_ID (0xffffffffffffffffULL)
typedef unsigned long long ull;

/* Stuffs that will be exposed to pybind11 */
class PlanNode;
std::vector<std::shared_ptr<PlanNode>> enumerate_plan(
    std::vector<std::vector<int>> adj_list, ull limit_num);

/*
 * Join-order-only plan nodes.
 *
 * A "plan" here is purely the join order: a binary tree whose leaves are
 * relations (identified by their index in the join graph) and whose internal
 * nodes are joins. No scan methods, no join methods — PostgreSQL decides those
 * after being handed a Leading() hint.
 */
class PlanNode {
 public:
  PlanNode() = default;
  virtual ~PlanNode() = default;
  virtual bool is_join() const = 0;
  virtual int relation() const { return -1; }
  virtual std::shared_ptr<PlanNode> left() const { return nullptr; }
  virtual std::shared_ptr<PlanNode> right() const { return nullptr; }
  friend std::ostream& operator<<(std::ostream& os, const PlanNode& node);
};

class Scan : public PlanNode {
 public:
  int relation_;
  explicit Scan(int relation) : relation_(relation) {}
  bool is_join() const override { return false; }
  int relation() const override { return relation_; }
};

class Join : public PlanNode {
 public:
  std::shared_ptr<PlanNode> left_, right_;
  Join(std::shared_ptr<PlanNode> left, std::shared_ptr<PlanNode> right)
      : left_(std::move(left)), right_(std::move(right)) {}
  bool is_join() const override { return true; }
  std::shared_ptr<PlanNode> left() const override { return left_; }
  std::shared_ptr<PlanNode> right() const override { return right_; }
};

class JoinGraph{
 public:
  ull adj_matrix_[MAX_RELATION_NUM];
  int vertex_num_;

  JoinGraph(int vertex_num, const std::vector<std::vector<int>> &adj_list);
  ~JoinGraph() = default;
  ull get_neighbors(int vertex_id) const;
  ull sample_subgraph(unsigned int size, ull *adj_matrix) const;
};


class PlanSpaceVertex {
 public:
  ull subquery_, id_, connected_;
  ull plan_num_;
  std::vector<std::pair<ull, ull>> child_;
  PlanSpaceVertex(ull subquery, ull id, ull connected);
  ~PlanSpaceVertex() = default;
  bool is_leaf() const;
  void add_child(ull left_psg_vertex_id, ull right_psg_vertex_id);
};

class PlanSpaceGraph{
 private:
  PlanSpaceGraph(std::vector<PlanSpaceVertex> vertices);

  void enumerate_plan(
      const PlanSpaceVertex &psg_vertex,
      std::vector<std::vector<std::shared_ptr<PlanNode>>> &plan_cache) const;

  std::vector<std::shared_ptr<PlanNode>> sample_plan(
      const PlanSpaceVertex &psg_vertex, ull sample_num,
      std::mt19937_64 &g) const;

 public:
  std::vector<PlanSpaceVertex> vertices_;
  const PlanSpaceVertex *root_;
  ull vertex_num_;

  /* For regular join graphs with #relations <= 16 */
  static PlanSpaceGraph from_join_graph(const JoinGraph &join_graph);

  /* For large join graphs that #relations > 16 */
  static PlanSpaceGraph sampled_plan_space(const JoinGraph &join_graph, ull sampled_plan_num);

  std::vector<std::shared_ptr<PlanNode>> enumerate_or_sample_plan(ull limit_num) const;
};
