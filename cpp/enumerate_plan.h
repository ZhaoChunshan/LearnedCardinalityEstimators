#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define MAX_RELATION_NUM 64
#define CONSTRUCT_PLAN_SPACE_LIMIT_SIZE 16
#define INVALID_SUBQUERY_ID (0xffffffffffffffffULL)
typedef unsigned long long ull;

enum class ScanMethod : int { SeqScan = 0, IndexScan = 1, IndexOnlyScan = 2, BitmapScan = 3 };
enum class JoinMethod : int { NestedLoopJoin = 0, HashJoin = 1, MergeJoin = 2 };

/* Stuffs that will be exposed to pybind11 */
class PlanNode;
std::vector<std::shared_ptr<PlanNode>> enumerate_plan(
    std::vector<std::vector<int>> adj_list, ull limit_num,
    const std::vector<std::string>& relation_names,
    const std::vector<std::vector<ScanMethod>>& supported_scan_methods,
    const std::vector<bool>& support_bitmap_as_nlj_right);

#define NUM_SCAN_METHODS 4
#define NUM_JOIN_METHODS 3

// Method vector helpers — source of truth for available methods
const std::vector<ScanMethod>& all_scan_methods();
const std::vector<JoinMethod>& all_join_methods();

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

// Concrete physical plan nodes — used only in final materialized output
class ScanNode : public PlanNode {
 public:
  int relation_;
  ScanMethod scan_method_;
  ScanNode(int relation, ScanMethod scan_method)
      : relation_(relation), scan_method_(scan_method) {}
  bool is_join() const override { return false; }
  int relation() const override { return relation_; }
  ScanMethod scan_method() const { return scan_method_; }
};

class JoinNode : public PlanNode {
 public:
  JoinMethod join_method_;
  std::shared_ptr<PlanNode> left_, right_;
  JoinNode(JoinMethod join_method, std::shared_ptr<PlanNode> left, std::shared_ptr<PlanNode> right)
      : join_method_(join_method), left_(std::move(left)), right_(std::move(right)) {}
  bool is_join() const override { return true; }
  std::shared_ptr<PlanNode> left() const override { return left_; }
  std::shared_ptr<PlanNode> right() const override { return right_; }
  JoinMethod join_method() const { return join_method_; }
};

// Plan-space template nodes — method vectors for undetermined execution choices
class ScanTemplate : public PlanNode {
 public:
  int relation_;
  std::vector<ScanMethod> scan_methods_;
  bool has_bitmap_bonus_; // true if BitmapScan was added (only valid as NLJ right child)
  ScanTemplate(int relation, std::vector<ScanMethod> methods, bool has_bitmap_bonus = false)
      : relation_(relation), scan_methods_(std::move(methods)),
        has_bitmap_bonus_(has_bitmap_bonus) {}
  bool is_join() const override { return false; }
  int relation() const override { return relation_; }
};

class JoinTemplate : public PlanNode {
 public:
  std::vector<JoinMethod> join_methods_;
  JoinTemplate(std::vector<JoinMethod> methods)
      : join_methods_(std::move(methods)) {}
  bool is_join() const override { return true; }
};

class JoinGraph{
 public:
  ull adj_matrix_[MAX_RELATION_NUM];
  int vertex_num_;
  std::vector<std::string> relation_names_;
  std::vector<std::vector<ScanMethod>> supported_scan_methods_;
  std::vector<bool> support_bitmap_as_nlj_right_;

  JoinGraph(int vertex_num, const std::vector<std::vector<int>> &adj_list,
            const std::vector<std::string> &relation_names,
            const std::vector<std::vector<ScanMethod>> &supported_scan_methods,
            const std::vector<bool> &support_bitmap_as_nlj_right);
  ~JoinGraph() = default;
  ull get_neighbors(int vertex_id) const;
  ull sample_subgraph(unsigned int size, ull *adj_matrix) const;
};


class PlanSpaceVertex {
 public:
  ull subquery_, id_, connected_;
  ull plan_num_;
  std::vector<std::pair<ull, ull>> child_;
  std::shared_ptr<PlanNode> plan_space_node_; // ScanTemplate or JoinTemplate
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
      std::mt19937_64 &g, bool is_nlj_right_child = false) const;

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
