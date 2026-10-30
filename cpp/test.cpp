#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "enumerate_plan.h"

// Helpers to construct default mappings for tests
static std::vector<std::string> make_names(int n) {
  std::vector<std::string> names;
  for (int i = 0; i < n; ++i) names.push_back("t" + std::to_string(i));
  return names;
}
static std::vector<std::vector<ScanMethod>> make_default_scan(int n) {
  return std::vector<std::vector<ScanMethod>>(n, all_scan_methods());
}
static std::vector<bool> make_no_bitmap_bonus(int n) {
  return std::vector<bool>(n, false);
}

/* Test Plan Space Graph */
bool test7(){
  ull subquery_num[6] = {0, 1, 3, 7, 15, 31};
  ull plan_num[6] = {0, 4, 96, 6912, 829440, 139345920};
  for(int size = 2; size <= 5; ++size){
    std::cout << "\tTest Clique of size " << size << std::endl;
    std::vector<std::vector<int>> adj_list(size);
    for(int i = 0; i < size; ++i){
      for(int j = 0; j < size; ++j){
        adj_list[i].push_back(j);
      }
    }
    JoinGraph join_graph(size, adj_list, make_names(size), make_default_scan(size), make_no_bitmap_bonus(size));
    PlanSpaceGraph plan_space_graph = PlanSpaceGraph::from_join_graph(join_graph);
  
    std::cout << "\tSubquery num: " << plan_space_graph.vertex_num_ << std::endl;
    std::cout << "\tPlan num: " << plan_space_graph.root_->plan_num_ << std::endl;
    if(plan_space_graph.root_->plan_num_ != plan_num[size] || plan_space_graph.vertex_num_ != subquery_num[size]){
      return false;
    }
  }
  return true;
}

bool test8() {
  std::vector<std::vector<int>> adj_list {
    {1, 2, 3},
    {0},
    {0},
    {0},
  };
  JoinGraph join_graph(4, adj_list, make_names(4), make_default_scan(4), make_no_bitmap_bonus(4));
  PlanSpaceGraph plan_space_graph = PlanSpaceGraph::from_join_graph(join_graph);
  std::cout << "\tTest star graph of size 4." << std::endl;
  std::cout << "\tSubquery num: " << plan_space_graph.vertex_num_ << std::endl;
  std::cout << "\tPlan num: " << plan_space_graph.root_->plan_num_ << std::endl;
  if(plan_space_graph.root_->plan_num_ != 331776 || plan_space_graph.vertex_num_ != 11)
    return false;
  return true;
}

// Sample subgraph
bool test9() {
  std::vector<std::vector<int>> adj_list {
    {1, 2}, {0, 2}, {0, 1, 3}, {2, 4}, {3, 5}, {4} 
  };
  JoinGraph g(6, adj_list, make_names(6), make_default_scan(6), make_no_bitmap_bonus(6));
  ull adj_matrix[MAX_RELATION_NUM];
  ull selected = g.sample_subgraph(3, adj_matrix);
  ull cnt = 0;
  while(selected){
    selected &= (selected - 1);
    cnt++;
  }
  return cnt == 3;
}

// Exaustive plan enumeration with methods
bool test10() {
  int size = 3;
  std::vector<std::vector<int>> adj_list(size);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      adj_list[i].push_back(j);
    }
  }
  JoinGraph join_graph(size, adj_list, make_names(size), make_default_scan(size), make_no_bitmap_bonus(size));
  PlanSpaceGraph plan_space_graph = PlanSpaceGraph::from_join_graph(join_graph);
  std::vector<std::shared_ptr<PlanNode>> plan =
      plan_space_graph.enumerate_or_sample_plan(1000000);  // large enough
  std::cout << "Enumerated " << plan.size() << " plans (expected 6912)" << std::endl;
  // Print a few representative plans
  for (int i = 0; i < 5 && i < (int)plan.size(); ++i) {
    std::cout << "  [" << i << "] " << *plan[i] << std::endl;
  }
  if (plan.size() != 6912) return false;
  // Verify every plan has valid method types
  for (auto& p : plan) {
    if (auto* jn = dynamic_cast<JoinNode*>(p.get())) {
      int jm = static_cast<int>(jn->join_method_);
      if (jm < 0 || jm >= NUM_JOIN_METHODS) return false;
    } else if (auto* sn = dynamic_cast<ScanNode*>(p.get())) {
      int sm = static_cast<int>(sn->scan_method_);
      if (sm < 0 || sm >= NUM_SCAN_METHODS) return false;
    }
  }
  return true;
}

// Sample plan space with methods
bool test11() {
  int size = 15;
  std::vector<std::vector<int>> adj_list(size);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      adj_list[i].push_back(j);
    }
  }
  JoinGraph join_graph(size, adj_list, make_names(size), make_default_scan(size), make_no_bitmap_bonus(size));
  PlanSpaceGraph plan_space_graph = PlanSpaceGraph::sampled_plan_space(join_graph, 5);
  auto plan_num = plan_space_graph.root_->plan_num_;
  std::cout << "#plans in sampled plan space: " << plan_num << std::endl;

  std::vector<std::shared_ptr<PlanNode>> plan = plan_space_graph.enumerate_or_sample_plan(5);
  for(auto& p: plan){
    std::cout << *p << std::endl;
  }
  if(plan.size() != 5) return false;
  // Verify sampled plans have valid method types
  for (auto& p : plan) {
    if (auto* jn = dynamic_cast<JoinNode*>(p.get())) {
      int jm = static_cast<int>(jn->join_method_);
      if (jm < 0 || jm >= NUM_JOIN_METHODS) return false;
    } else if (auto* sn = dynamic_cast<ScanNode*>(p.get())) {
      int sm = static_cast<int>(sn->scan_method_);
      if (sm < 0 || sm >= NUM_SCAN_METHODS) return false;
    }
  }
  return true;
}


// Efficiency test (DPsize not good for large clique/star)
bool efficiency_test(int size = 15){
  std::vector<std::vector<int>> adj_list(size);
  for(int i = 0; i < size; ++i){
    for(int j = 0; j < size; ++j){
      adj_list[i].push_back(j);
    }
  }
  JoinGraph join_graph(size, adj_list, make_names(size), make_default_scan(size), make_no_bitmap_bonus(size));
  PlanSpaceGraph plan_space_graph = PlanSpaceGraph::from_join_graph(join_graph);
  
  std::cout << "\tSubquery num: " << plan_space_graph.vertex_num_ << std::endl;
  std::cout << "\tPlan num: " << plan_space_graph.root_->plan_num_ << std::endl;
  return true;
}

int main(){
  bool (*tests[])() = {test7, test8, test9, test10, test11};
  for (int i = 0; i < (int)(sizeof(tests) / sizeof(tests[0])); i++){
    std::cout << "\033[1;32m" << "Execute test case #" << i + 1 << "..."
              << "\033[0m" << std::endl;
    assert(tests[i]());
  }
  std::cout << "\033[1;32m" << "All test cases passed." << "\033[0m" << std::endl;
  return 0;
}
