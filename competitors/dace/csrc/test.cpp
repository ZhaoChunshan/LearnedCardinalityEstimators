#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "enumerate_plan.h"

// Make a full clique adjacency (self-loops included) of the given size.
static std::vector<std::vector<int>> make_clique(int size) {
  std::vector<std::vector<int>> adj_list(size);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      adj_list[i].push_back(j);
    }
  }
  return adj_list;
}

/* Test Plan Space Graph: join-order counts for cliques.
 * For a clique of n relations, the number of ordered bushy join trees with
 * labeled leaves f(n) satisfies f(1)=1 and f(n)=sum C(n,k) f(k) f(n-k):
 *   n=1 -> 1, n=2 -> 2, n=3 -> 12, n=4 -> 120, n=5 -> 1680.
 */
bool test1(){
  ull subquery_num[6] = {0, 1, 3, 7, 15, 31};
  ull plan_num[6] = {0, 1, 2, 12, 120, 1680};
  for(int size = 2; size <= 5; ++size){
    std::cout << "\tTest Clique of size " << size << std::endl;
    JoinGraph join_graph(size, make_clique(size));
    PlanSpaceGraph plan_space_graph = PlanSpaceGraph::from_join_graph(join_graph);

    std::cout << "\tSubquery num: " << plan_space_graph.vertex_num_ << std::endl;
    std::cout << "\tPlan num: " << plan_space_graph.root_->plan_num_ << std::endl;
    if(plan_space_graph.root_->plan_num_ != plan_num[size] || plan_space_graph.vertex_num_ != subquery_num[size]){
      return false;
    }
  }
  return true;
}

/* Star graph: center 0 joined to leaves 1,2,3. Valid join orders count to 48:
 * each of the 3 ordered "center vs leaves" root splits contributes 16
 * (2 orientations x 8 join orders inside the leaves subtree). */
bool test2() {
  std::vector<std::vector<int>> adj_list {
    {1, 2, 3},
    {0},
    {0},
    {0},
  };
  JoinGraph join_graph(4, adj_list);
  PlanSpaceGraph plan_space_graph = PlanSpaceGraph::from_join_graph(join_graph);
  std::cout << "\tTest star graph of size 4." << std::endl;
  std::cout << "\tSubquery num: " << plan_space_graph.vertex_num_ << std::endl;
  std::cout << "\tPlan num: " << plan_space_graph.root_->plan_num_ << std::endl;
  if(plan_space_graph.root_->plan_num_ != 48 || plan_space_graph.vertex_num_ != 11)
    return false;
  return true;
}

// Sample subgraph
bool test3() {
  std::vector<std::vector<int>> adj_list {
    {1, 2}, {0, 2}, {0, 1, 3}, {2, 4}, {3, 5}, {4}
  };
  JoinGraph g(6, adj_list);
  ull adj_matrix[MAX_RELATION_NUM];
  ull selected = g.sample_subgraph(3, adj_matrix);
  ull cnt = 0;
  while(selected){
    selected &= (selected - 1);
    cnt++;
  }
  return cnt == 3;
}

// Verify a join-order tree is well-formed: leaves are Scans covering exactly
// the relations of the subtree (bitmask), internal nodes are Joins.
static bool check_plan_shape(const PlanNode *node, ull &relations, int expected_relations) {
  if (auto *sn = dynamic_cast<const Scan*>(node)) {
    if (sn->relation() < 0 || sn->relation() >= expected_relations) return false;
    relations |= (1ULL << sn->relation());
    return true;
  }
  auto *jn = dynamic_cast<const Join*>(node);
  if (!jn) return false;
  ull l = 0, r = 0;
  if (!check_plan_shape(jn->left_.get(), l, expected_relations)) return false;
  if (!check_plan_shape(jn->right_.get(), r, expected_relations)) return false;
  if (l & r) return false;  // overlapping relation sets
  relations = l | r;
  return true;
}

// Exhaustive join-order enumeration
bool test4() {
  int size = 3;
  JoinGraph join_graph(size, make_clique(size));
  PlanSpaceGraph plan_space_graph = PlanSpaceGraph::from_join_graph(join_graph);
  std::vector<std::shared_ptr<PlanNode>> plan =
      plan_space_graph.enumerate_or_sample_plan(1000000);  // large enough
  std::cout << "Enumerated " << plan.size() << " join orders (expected 12)" << std::endl;
  for (int i = 0; i < 5 && i < (int)plan.size(); ++i) {
    std::cout << "  [" << i << "] " << *plan[i] << std::endl;
  }
  if (plan.size() != 12) return false;
  // Every plan must be a valid tree covering all 3 relations exactly once.
  for (auto& p : plan) {
    ull relations = 0;
    if (!check_plan_shape(p.get(), relations, size) || relations != 0b111ULL)
      return false;
  }
  return true;
}

// Sample plan space
bool test5() {
  int size = 15;
  JoinGraph join_graph(size, make_clique(size));
  PlanSpaceGraph plan_space_graph = PlanSpaceGraph::sampled_plan_space(join_graph, 5);
  auto plan_num = plan_space_graph.root_->plan_num_;
  std::cout << "#plans in sampled plan space: " << plan_num << std::endl;

  std::vector<std::shared_ptr<PlanNode>> plan = plan_space_graph.enumerate_or_sample_plan(5);
  for(auto& p: plan){
    std::cout << *p << std::endl;
  }
  if(plan.size() != 5) return false;
  // Verify sampled plans are well-formed and cover all relations.
  for (auto& p : plan) {
    ull relations = 0;
    if (!check_plan_shape(p.get(), relations, size) || relations != ((1ULL << size) - 1))
      return false;
  }
  return true;
}


// Efficiency test (DPsize not good for large clique/star)
bool efficiency_test(int size = 15){
  JoinGraph join_graph(size, make_clique(size));
  PlanSpaceGraph plan_space_graph = PlanSpaceGraph::from_join_graph(join_graph);

  std::cout << "\tSubquery num: " << plan_space_graph.vertex_num_ << std::endl;
  std::cout << "\tPlan num: " << plan_space_graph.root_->plan_num_ << std::endl;
  return true;
}

int main(){
  bool (*tests[])() = {test1, test2, test3, test4, test5};
  for (int i = 0; i < (int)(sizeof(tests) / sizeof(tests[0])); i++){
    std::cout << "\033[1;32m" << "Execute test case #" << i + 1 << "..."
              << "\033[0m" << std::endl;
    assert(tests[i]());
  }
  std::cout << "\033[1;32m" << "All test cases passed." << "\033[0m" << std::endl;
  return 0;
}
