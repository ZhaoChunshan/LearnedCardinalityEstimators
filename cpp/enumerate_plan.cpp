#include <array>
#include <climits>
#include <numeric>
#include <random>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "enumerate_plan.h"

std::vector<std::shared_ptr<PlanNode>> enumerate_plan(
    std::vector<std::vector<int>> adj_list, ull limit_num,
    const std::vector<std::string> &relation_names,
    const std::vector<std::vector<ScanMethod>> &supported_scan_methods,
    const std::vector<bool> &support_bitmap_as_nlj_right)
{
  int vertex_num_ = (int)adj_list.size();
  if(vertex_num_ > MAX_RELATION_NUM){
    std::cerr << "[CPP] Too many relations: " << vertex_num_ << std::endl;
    exit(1);
  }

  JoinGraph join_graph(vertex_num_, adj_list, relation_names,
                       supported_scan_methods, support_bitmap_as_nlj_right);
  if(vertex_num_ <= CONSTRUCT_PLAN_SPACE_LIMIT_SIZE){
    PlanSpaceGraph plan_space_graph = PlanSpaceGraph::from_join_graph(join_graph);
    return plan_space_graph.enumerate_or_sample_plan(limit_num);
  }
  else{
    PlanSpaceGraph plan_space_graph = PlanSpaceGraph::sampled_plan_space(join_graph, limit_num);
    return plan_space_graph.enumerate_or_sample_plan(limit_num);
  }
  return {};
}

static const char* scan_method_names[] = {"SeqScan", "IndexScan", "IndexOnlyScan", "BitmapScan"};
static const char* join_method_names[] = {"NestedLoopJoin", "HashJoin", "MergeJoin"};

const std::vector<ScanMethod>& all_scan_methods() {
  static const std::vector<ScanMethod> methods = {
      ScanMethod::SeqScan, ScanMethod::IndexScan,
      ScanMethod::IndexOnlyScan, ScanMethod::BitmapScan};
  return methods;
}

const std::vector<JoinMethod>& all_join_methods() {
  static const std::vector<JoinMethod> methods = {
      JoinMethod::NestedLoopJoin, JoinMethod::HashJoin, JoinMethod::MergeJoin};
  return methods;
}

std::ostream& operator<<(std::ostream &os, const PlanNode &node) {
  if (auto* jn = dynamic_cast<const JoinNode*>(&node)) {
    os << join_method_names[static_cast<int>(jn->join_method_)] << "("
       << *(jn->left_) << ", " << *(jn->right_) << ")";
  } else if (auto* sn = dynamic_cast<const ScanNode*>(&node)) {
    os << scan_method_names[static_cast<int>(sn->scan_method_)] << "("
       << sn->relation_ << ")";
  } else if (dynamic_cast<const JoinTemplate*>(&node)) {
    os << "JoinTemplate";
  } else if (auto* st = dynamic_cast<const ScanTemplate*>(&node)) {
    os << "ScanTemplate(" << st->relation_ << ")";
  }
  return os;
}

JoinGraph::JoinGraph(int vertex_num, const std::vector<std::vector<int>> &adj_list,
                     const std::vector<std::string> &relation_names,
                     const std::vector<std::vector<ScanMethod>> &supported_scan_methods,
                     const std::vector<bool> &support_bitmap_as_nlj_right)
    : vertex_num_(vertex_num), relation_names_(relation_names),
      supported_scan_methods_(supported_scan_methods),
      support_bitmap_as_nlj_right_(support_bitmap_as_nlj_right) {
  assert((int)adj_list.size() == vertex_num);
  assert((int)relation_names.size() == vertex_num);
  assert((int)supported_scan_methods.size() == vertex_num);
  assert((int)support_bitmap_as_nlj_right.size() == vertex_num);
  std::memset(adj_matrix_, 0, sizeof(ull) * MAX_RELATION_NUM);
  for(int i = 0; i < vertex_num; ++i){
    for(auto j: adj_list[i]){
      adj_matrix_[i] |= (1ULL << j);
    }
  }
}

ull JoinGraph::get_neighbors(int vertex_id) const {
  return adj_matrix_[vertex_id];
}

ull JoinGraph::sample_subgraph(unsigned int size, ull *adj_matrix) const {
  ull selected_vertices = 0ULL;

  std::vector<int> queue;
  queue.reserve(vertex_num_);
  queue.push_back(0);
  selected_vertices |= (1ULL << 0);
  unsigned int i = 0;

  for(; queue.size() < size; ++i){
    int v = queue[i];
    ull neighbors = adj_matrix_[v];
    while(neighbors){
      int u = __builtin_ctzll(neighbors);
      neighbors &= ~(1ULL << u);
      if(!(selected_vertices & (1ULL << u))){
        queue.push_back(u);
        selected_vertices |= (1ULL << u);
        if(queue.size() == size) break;
      }
    }
  }

  memcpy(adj_matrix, adj_matrix_, sizeof(ull) * MAX_RELATION_NUM);
  for(int i = 0; i < vertex_num_; ++i){
    if(!(selected_vertices & (1ULL << i))){
      adj_matrix[i] = 0;
    }
    else {
      adj_matrix[i] = adj_matrix_[i] & selected_vertices;
    }
  }

  return selected_vertices;
}


PlanSpaceVertex::PlanSpaceVertex(ull subquery, ull id, ull connected): subquery_(subquery), id_(id), connected_(connected), plan_num_(0) {

}

bool PlanSpaceVertex::is_leaf() const{
  return child_.size() == 0;
}

void PlanSpaceVertex::add_child(ull left_psg_vertex_id, ull right_psg_vertex_id){
  child_.emplace_back(left_psg_vertex_id, right_psg_vertex_id);
}

PlanSpaceGraph::PlanSpaceGraph(std::vector<PlanSpaceVertex> vertices): vertices_(std::move(vertices)) {
  root_ = &(vertices_.back());
  vertex_num_ = (int)vertices_.size();
}

PlanSpaceGraph PlanSpaceGraph::from_join_graph(const JoinGraph &join_graph){
  int vertex_num = join_graph.vertex_num_;

  std::vector<PlanSpaceVertex> vertices;
  std::vector<ull> subquery_2_id(1 << vertex_num, INVALID_SUBQUERY_ID);
  std::vector<std::vector<ull>> size_subquery(vertex_num + 1, std::vector<ull>());
  vertices.reserve(1 << vertex_num);

  /* Subquery with only 1 relation */
  for(int v = 0; v < vertex_num; ++v){
    ull id = vertices.size();
    ull subquery = 1ULL << v;
    ull connected = join_graph.get_neighbors(v);
    vertices.emplace_back(subquery, id, connected);
    // Build scan methods: optionally append BitmapScan as NLJ-right-child bonus
    std::vector<ScanMethod> methods = join_graph.supported_scan_methods_[v];
    bool has_bonus = false;
    if (join_graph.support_bitmap_as_nlj_right_[v]) {
      bool already_has_bitmap = false;
      for (auto m : methods)
        if (m == ScanMethod::BitmapScan) { already_has_bitmap = true; break; }
      if (!already_has_bitmap) {
        methods.push_back(ScanMethod::BitmapScan);
        has_bonus = true;
      }
    }
    auto scan_tmpl = std::make_shared<ScanTemplate>(v, std::move(methods), has_bonus);
    vertices.back().plan_space_node_ = scan_tmpl;
    vertices.back().plan_num_ = scan_tmpl->scan_methods_.size();
    subquery_2_id[subquery] = id;
    size_subquery[1].push_back(id);
  }

  /* Handle Joins, Called By DPsize */
  auto handle_join_pairs = [&vertices, &subquery_2_id, &size_subquery](
                               ull left_id, ull right_id, int size) {
    
    bool left_bonus = false, right_bonus = false;
    if (vertices[left_id].is_leaf()) {
      auto& l_scan =
          static_cast<const ScanTemplate&>(*vertices[left_id].plan_space_node_);
      if (l_scan.has_bitmap_bonus_) {
        left_bonus = true;
      }
    }
    if (vertices[right_id].is_leaf()) {
      auto& r_scan = static_cast<const ScanTemplate&>(
          *vertices[right_id].plan_space_node_);
      if (r_scan.has_bitmap_bonus_) {
        right_bonus = true;
      }
    }
    
    ull subquery = vertices[left_id].subquery_ | vertices[right_id].subquery_;
    ull left_num = left_bonus ? (vertices[left_id].plan_num_ - 1) : vertices[left_id].plan_num_;
    ull right_num = vertices[right_id].plan_num_;
    ull join_cnt = NUM_JOIN_METHODS;
    
    ull contribution = left_num * right_num * join_cnt;

    if(right_bonus){
      contribution -= left_num * (join_cnt - 1);
    }

    ull id = subquery_2_id[subquery];
    if(id != INVALID_SUBQUERY_ID){
      auto& psg_vertex = vertices[id];
      psg_vertex.plan_num_ += contribution;
      psg_vertex.add_child(left_id, right_id);
      return;
    }
    else {
      id = vertices.size();
      ull connected = (vertices[left_id].connected_ | vertices[right_id].connected_) & (~subquery);
      vertices.emplace_back(subquery, id, connected);
      auto join_tmpl = std::make_shared<JoinTemplate>(all_join_methods());
      vertices.back().plan_space_node_ = join_tmpl;
      vertices.back().plan_num_ = contribution;
      vertices.back().add_child(left_id, right_id);
      subquery_2_id[subquery] = id;
      size_subquery[size].push_back(id);
    }
  };

  /* DPsize algorithm, easier to implement than DPccp but works well */
  for(int size = 2; size <= vertex_num; ++size){
    for(int s1 = 1; (s1 << 1) < size; ++s1){ /* assume s1 < s2 */
      int s2 = size - s1;
      for(ull left_id : size_subquery[s1]){
        for(ull right_id : size_subquery[s2]){
          if(vertices[left_id].subquery_ & vertices[right_id].subquery_ /* overlap*/ || 
            !(vertices[left_id].connected_ & vertices[right_id].subquery_) /* disjoint */){
            continue;
          }
          handle_join_pairs(left_id, right_id, size);
          handle_join_pairs(right_id, left_id, size); 
        }
      }
    }
    if(!(size & 1)){
      int half = size >> 1;
      for(int i = 0; i < (int)size_subquery[half].size(); ++i){
        for(int j = i + 1; j < (int)size_subquery[half].size(); ++j){
          ull left_id = size_subquery[half][i];
          ull right_id = size_subquery[half][j];
          if(vertices[left_id].subquery_ & vertices[right_id].subquery_ /* overlap*/ ||
            !(vertices[left_id].connected_ & vertices[right_id].subquery_) /* disjoint */){
            continue;
          }
          handle_join_pairs(left_id, right_id, size);
          handle_join_pairs(right_id, left_id, size); 
        }
      }
    }
  }

  return PlanSpaceGraph(vertices);
}

PlanSpaceGraph PlanSpaceGraph::sampled_plan_space(const JoinGraph &join_graph,
                                                  ull sampled_plan_num)
{
  /**
   * A subgraph with size 16 is sufficient to meet a moderate # of plans.
   * For a join graph with size n, the minimum number of join order is
   * 2^{n-1}*catalan(n-1) (chain join), which is 317,680,680,960 for n=16. We can
   * sample a subgraph with size 16 to construct the plan space, and then
   * connect the remaining vertices one by one to get the final plan space.
   */
  unsigned int subgraph_size = std::min((unsigned int)join_graph.vertex_num_, (unsigned int)CONSTRUCT_PLAN_SPACE_LIMIT_SIZE);
  int vertex_num = join_graph.vertex_num_;

  assert(sampled_plan_num <= 317680680960ULL); // 2^{15}*catalan(15)
  
  ull adj_matrix[MAX_RELATION_NUM];
  ull sampled_vertices = join_graph.sample_subgraph(subgraph_size, adj_matrix);

  std::vector<PlanSpaceVertex> vertices;
  std::unordered_map<ull, ull> subquery_2_id;
  std::vector<std::vector<ull>> size_subquery(vertex_num + 1,
                                              std::vector<ull>());
  vertices.reserve((1 << subgraph_size) + (vertex_num << 1));

  /* Subquery with only 1 relation */
  for (int v = 0; v < vertex_num; ++v) { // We must not skip the vertex not contained in the sampled subgraph
                                         // Since they are used later
    ull id = vertices.size();
    ull subquery = 1ULL << v;
    ull connected = adj_matrix[v];
    vertices.emplace_back(subquery, id, connected);
    std::vector<ScanMethod> methods = join_graph.supported_scan_methods_[v];
    bool has_bonus = false;
    if (join_graph.support_bitmap_as_nlj_right_[v]) {
      bool already_has_bitmap = false;
      for (auto m : methods)
        if (m == ScanMethod::BitmapScan) { already_has_bitmap = true; break; }
      if (!already_has_bitmap) { methods.push_back(ScanMethod::BitmapScan); has_bonus = true; }
    }
    auto scan_tmpl = std::make_shared<ScanTemplate>(v, std::move(methods), has_bonus);
    vertices.back().plan_space_node_ = scan_tmpl;
    vertices.back().plan_num_ = scan_tmpl->scan_methods_.size();
    subquery_2_id[subquery] = id;
    size_subquery[1].push_back(id);
  }

  /* Handle Joins, Called By DPsize */
  auto handle_join_pairs = [&vertices, &subquery_2_id, &size_subquery](
                               ull left_id, ull right_id, int size) {
    
    bool left_bonus = false, right_bonus = false;
    if (vertices[right_id].is_leaf()) {
      auto& r_scan = static_cast<const ScanTemplate&>(*vertices[right_id].plan_space_node_);
      if (r_scan.has_bitmap_bonus_) {
        right_bonus = true;
      }
    }
    if (vertices[left_id].is_leaf()){
      auto& l_scan = static_cast<const ScanTemplate&>(*vertices[left_id].plan_space_node_);
      if (l_scan.has_bitmap_bonus_) {
        left_bonus = true;
      }
    }

    ull subquery = vertices[left_id].subquery_ | vertices[right_id].subquery_;
    ull left_num = left_bonus ? (vertices[left_id].plan_num_ - 1) : vertices[left_id].plan_num_;
    ull right_num = vertices[right_id].plan_num_;
    ull join_cnt = NUM_JOIN_METHODS;

    ull contribution = left_num * right_num * join_cnt;

    if(right_bonus){
      contribution -= left_num * (join_cnt - 1);
    }

    auto it = subquery_2_id.find(subquery);
    if (it != subquery_2_id.end()) {
      ull id = it->second;
      auto &psg_vertex = vertices[id];
      psg_vertex.plan_num_ += contribution;
      psg_vertex.add_child(left_id, right_id);
    } else {
      ull id = vertices.size();
      ull connected =
          (vertices[left_id].connected_ | vertices[right_id].connected_) &
          (~subquery);
      vertices.emplace_back(subquery, id, connected);
      auto join_tmpl = std::make_shared<JoinTemplate>(all_join_methods());
      vertices.back().plan_space_node_ = join_tmpl;
      vertices.back().plan_num_ = contribution;
      vertices.back().add_child(left_id, right_id);
      subquery_2_id[subquery] = id;
      size_subquery[size].push_back(id);
    }
    return;
  };

  /* DPsize algorithm, easier to implement than DPccp but works well */
  for (int size = 2; size <= (int)subgraph_size; ++size) {
    for (int s1 = 1; (s1 << 1) < size; ++s1) { /* assume s1 < s2 */
      int s2 = size - s1;
      for (ull left_id : size_subquery[s1]) {
        for (ull right_id : size_subquery[s2]) {
          if (vertices[left_id].subquery_ &
                  vertices[right_id].subquery_ /* overlap*/
              || !(vertices[left_id].connected_ &
                   vertices[right_id].subquery_) /* disjoint */) {
            continue;
          }
          handle_join_pairs(left_id, right_id, size);
          handle_join_pairs(right_id, left_id, size); 
        }
      }
    }
    if (!(size & 1)) {
      int half = size >> 1;
      for (int i = 0; i < (int)size_subquery[half].size(); ++i) {
        for (int j = i + 1; j < (int)size_subquery[half].size(); ++j) {
          ull left_id = size_subquery[half][i];
          ull right_id = size_subquery[half][j];
          if (vertices[left_id].subquery_ &
                  vertices[right_id].subquery_ /* overlap*/
              || !(vertices[left_id].connected_ &
                   vertices[right_id].subquery_) /* disjoint */) {
            continue;
          }
          handle_join_pairs(left_id, right_id, size);
          handle_join_pairs(right_id, left_id, size); 
        }
      }
    }
  }

  ull connected = 0;
  for (int v = 0; v < vertex_num; ++v) {
    if (!(sampled_vertices & (1ULL << v))) continue;
    connected |= join_graph.get_neighbors(v);
  }
  connected &= ~sampled_vertices;

  ull remain = ((1ULL << vertex_num) - 1) & ~sampled_vertices;
  while(remain){
    remain &= ~connected;
    ull next_connected = 0;
    while(connected){
      int v = __builtin_ctzll(connected);
      connected &= ~(1ULL << v);
      next_connected |= join_graph.get_neighbors(v) & remain;
      ull left_id = vertices.back().id_, right_id = subquery_2_id[1ULL << v];
      int size = __builtin_popcountll((vertices.back().subquery_) | (1ULL << v));
      handle_join_pairs(left_id, right_id, size); 
      handle_join_pairs(right_id, left_id, size);
    }
    connected = next_connected;
  }

  return PlanSpaceGraph(vertices);
}

void PlanSpaceGraph::enumerate_plan(
    const PlanSpaceVertex &psg_vertex,
    std::vector<std::vector<std::shared_ptr<PlanNode>>> &plan_cache) const
{
  if(psg_vertex.is_leaf()){
    auto& scan_tmpl = static_cast<const ScanTemplate&>(*psg_vertex.plan_space_node_);
    for (auto method : scan_tmpl.scan_methods_) {
      plan_cache[psg_vertex.id_].push_back(
          std::make_shared<ScanNode>(scan_tmpl.relation_, method));
    }
    return;
  }

  auto &plans = plan_cache[psg_vertex.id_];
  plans.reserve(psg_vertex.plan_num_);

  for(auto [left_id, right_id] : psg_vertex.child_){
    if(plan_cache[left_id].empty()){
      enumerate_plan(vertices_[left_id], plan_cache);
    }
    if(plan_cache[right_id].empty()){
      enumerate_plan(vertices_[right_id], plan_cache);
    }
    auto &left = plan_cache[left_id];
    auto &right = plan_cache[right_id];

    // Right-child BitmapScan bonus: only valid with NestedLoopJoin
    bool left_has_bonus = false, right_has_bonus = false;
    if (vertices_[left_id].is_leaf()) {
      left_has_bonus = static_cast<const ScanTemplate&>(
          *vertices_[left_id].plan_space_node_).has_bitmap_bonus_;
    }
    if (vertices_[right_id].is_leaf()) {
      right_has_bonus = static_cast<const ScanTemplate&>(
          *vertices_[right_id].plan_space_node_).has_bitmap_bonus_;
    }

    auto& join_tmpl = static_cast<const JoinTemplate&>(*psg_vertex.plan_space_node_);
    for(const auto &l : left){
      for(const auto &r : right){
        for (auto method : join_tmpl.join_methods_) {
          if (method != JoinMethod::NestedLoopJoin && right_has_bonus) {
            auto* sn = dynamic_cast<const ScanNode*>(r.get());
            if (sn && sn->scan_method_ == ScanMethod::BitmapScan) continue;
          }
          if (left_has_bonus){
            auto* sn = dynamic_cast<const ScanNode*>(l.get());
            if (sn && sn->scan_method_ == ScanMethod::BitmapScan) continue;
          }
          plans.emplace_back(std::make_shared<JoinNode>(method, l, r));
        }
      }
    }
  }
}

std::vector<std::shared_ptr<PlanNode>> PlanSpaceGraph::sample_plan(
    const PlanSpaceVertex &psg_vertex, ull sample_num,
    std::mt19937_64 &g, bool is_nlj_right_child) const
{
  if (sample_num == 0) return {};

  // Leaf node: Scan
  if (psg_vertex.is_leaf()) {
    auto &tmpl = static_cast<const ScanTemplate&>(*psg_vertex.plan_space_node_);

    std::vector<ScanMethod> valid_methods = tmpl.scan_methods_;
    if (tmpl.has_bitmap_bonus_ && !is_nlj_right_child) {
      valid_methods.erase(std::remove(valid_methods.begin(), valid_methods.end(), ScanMethod::BitmapScan), valid_methods.end());
    }

    ull n = valid_methods.size();
    assert(sample_num <= n);

    std::vector<std::shared_ptr<PlanNode>> result;
    result.reserve(sample_num);

    // Sample without replacement: shuffle indices, take first sample_num
    std::vector<ull> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), g);
    for (ull i = 0; i < sample_num; ++i)
      result.emplace_back(std::make_shared<ScanNode>(tmpl.relation_,
          valid_methods[idx[i]]));

    return result;
  }

  // Internal node: Join

  auto &join_tmpl = static_cast<const JoinTemplate&>(*psg_vertex.plan_space_node_);
  const ull num_join_methods = join_tmpl.join_methods_.size();
  const size_t num_pairs = psg_vertex.child_.size();

  std::vector<ull> left_id(num_pairs), right_id(num_pairs), pair_capacity(num_pairs);
  std::vector<bool> left_bonus(num_pairs, false), right_bonus(num_pairs, false);
  ull total_capacity = 0;

  for (size_t i = 0; i < num_pairs; ++i) {
    auto [l, r] = psg_vertex.child_[i];
    left_id[i] = l;  right_id[i] = r;

    if (vertices_[r].is_leaf()) {
      auto &rs = static_cast<const ScanTemplate&>(*vertices_[r].plan_space_node_);
      if (rs.has_bitmap_bonus_) {
        right_bonus[i] = true;
      }
    }
    if (vertices_[l].is_leaf()) {
      auto &ls = static_cast<const ScanTemplate&>(*vertices_[l].plan_space_node_);
      if (ls.has_bitmap_bonus_) {
        left_bonus[i] = true;
      }
    }

    ull left_num = left_bonus[i] ? (vertices_[l].plan_num_ - 1) : vertices_[l].plan_num_;
    ull right_num = vertices_[r].plan_num_;

    ull capacity = left_num * right_num * num_join_methods;

    if(right_bonus[i]){
      capacity -= left_num * (num_join_methods - 1);
    }

    pair_capacity[i] = capacity;
    total_capacity += capacity;
  }

  if (total_capacity == 0) return {}; 

  std::vector<ull> sample_num_per_pair(num_pairs, 0);
  ull remain_sample = sample_num;
  ull remain_capacity = total_capacity;

  assert(remain_sample <= remain_capacity);

  for (size_t i = 0; i < num_pairs && remain_sample > 0; ++i) {
    if (pair_capacity[i] == 0) continue;
    ull cur_num = 0;
    if (i + 1 == num_pairs) {
      assert(remain_capacity >= remain_sample);
      assert(remain_capacity == pair_capacity[i]);
      cur_num = remain_sample;
    } else {
      double p = (double)pair_capacity[i] / (double)remain_capacity;
      if (p > 1.0) p = 1.0;
      cur_num = std::binomial_distribution<ull>(remain_sample, p)(g);
      if (cur_num > pair_capacity[i]) cur_num = pair_capacity[i];
      if (remain_sample - cur_num > remain_capacity - pair_capacity[i]) {
        cur_num = remain_sample - (remain_capacity - pair_capacity[i]);
      }
    }
    sample_num_per_pair[i] = cur_num;
    remain_sample -= cur_num;
    remain_capacity -= pair_capacity[i];
  }

  assert(std::accumulate(sample_num_per_pair.begin(), sample_num_per_pair.end(), 0ULL) == sample_num);

  std::vector<std::shared_ptr<PlanNode>> result;
  result.reserve(sample_num);

  for (size_t pi = 0; pi < num_pairs; ++pi) {
    ull cur_num = sample_num_per_pair[pi];
    if (cur_num == 0) continue;

    ull lid = left_id[pi], rid = right_id[pi];
    ull left_num = left_bonus[pi] ? (vertices_[lid].plan_num_ - 1) : vertices_[lid].plan_num_;
    ull right_num = vertices_[rid].plan_num_;

    std::vector<ull> method_capacity(num_join_methods, 0);
    ull total_method_capacity = 0;
    for (ull m = 0; m < num_join_methods; ++m) {
      ull c = left_num * right_num;
      if (right_bonus[pi] && join_tmpl.join_methods_[m] != JoinMethod::NestedLoopJoin)
        c -= left_num;
      
      method_capacity[m] = c;
      total_method_capacity += c;
    }

    ull remain_sample_m = cur_num;
    ull remain_capacity_m = total_method_capacity;
    for (ull m = 0; m < num_join_methods && remain_sample_m > 0; ++m) {
      if (method_capacity[m] == 0) continue;
      ull cur_num_m = 0;
      if (m + 1 == num_join_methods) {
        assert(remain_capacity_m >= remain_sample_m);
        assert(remain_capacity_m == method_capacity[m]);
        cur_num_m = remain_sample_m;
      } else {
        double p = (double)method_capacity[m] / (double)remain_capacity_m;
        if (p > 1.0) p = 1.0;
        cur_num_m = std::binomial_distribution<ull>(remain_sample_m, p)(g);
        if (cur_num_m > method_capacity[m]) cur_num_m = method_capacity[m];
        if (remain_sample_m - cur_num_m > remain_capacity_m - method_capacity[m]) {
          cur_num_m = remain_sample_m - (remain_capacity_m - method_capacity[m]);
        }
      }
      remain_sample_m -= cur_num_m;
      remain_capacity_m -= method_capacity[m];
      if (cur_num_m == 0) continue;

      bool is_nlj = (join_tmpl.join_methods_[m] == JoinMethod::NestedLoopJoin);
      auto left_sample_plan_num = std::min(left_num, cur_num_m);
      auto right_sample_plan_num = std::min(right_num, cur_num_m);
      if (!is_nlj && right_bonus[pi]) {
        right_sample_plan_num = std::min(right_num - 1, cur_num_m);
      }

      ull minimum_right_plan_required = (cur_num_m + left_sample_plan_num - 1) / left_sample_plan_num;
      right_sample_plan_num = std::min(right_sample_plan_num, minimum_right_plan_required);

      auto left_plans =
          sample_plan(vertices_[lid], left_sample_plan_num, g, false);
      auto right_plans =
          sample_plan(vertices_[rid], right_sample_plan_num, g, is_nlj);

      JoinMethod method = join_tmpl.join_methods_[m];
      ull cnt = 0;
      for(auto &l : left_plans){
        if(cnt == cur_num_m) break;
        for(auto &r : right_plans){
          result.emplace_back(std::make_shared<JoinNode>(method, l, r));
          if(++cnt == cur_num_m) break;
        }
      }
    }
  }

  return result;
}

std::vector<std::shared_ptr<PlanNode>> PlanSpaceGraph::enumerate_or_sample_plan(
    ull limit_num) const
{
  if (root_->plan_num_ <= limit_num) {
    std::vector<std::vector<std::shared_ptr<PlanNode>>> plan_cache(
        vertex_num_, std::vector<std::shared_ptr<PlanNode>>());
    enumerate_plan(*root_, plan_cache);
    return std::move(plan_cache[root_->id_]);
  }
  // std::random_device rd;
  std::mt19937_64 g(42);  // enable reproducibility
  return sample_plan(*root_, limit_num, g);
}
