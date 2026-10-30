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
    std::vector<std::vector<int>> adj_list, ull limit_num)
{
  int vertex_num_ = (int)adj_list.size();
  if(vertex_num_ > MAX_RELATION_NUM){
    std::cerr << "[CPP] Too many relations: " << vertex_num_ << std::endl;
    exit(1);
  }

  JoinGraph join_graph(vertex_num_, adj_list);
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

std::ostream& operator<<(std::ostream &os, const PlanNode &node) {
  if (auto* jn = dynamic_cast<const Join*>(&node)) {
    os << "Join(" << *(jn->left_) << ", " << *(jn->right_) << ")";
  } else if (auto* sn = dynamic_cast<const Scan*>(&node)) {
    os << "Scan(" << sn->relation_ << ")";
  }
  return os;
}

JoinGraph::JoinGraph(int vertex_num, const std::vector<std::vector<int>> &adj_list)
    : vertex_num_(vertex_num) {
  assert((int)adj_list.size() == vertex_num);
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

  /* Subquery with only 1 relation: exactly one join order (a single Scan) */
  for(int v = 0; v < vertex_num; ++v){
    ull id = vertices.size();
    ull subquery = 1ULL << v;
    ull connected = join_graph.get_neighbors(v);
    vertices.emplace_back(subquery, id, connected);
    vertices.back().plan_num_ = 1;
    subquery_2_id[subquery] = id;
    size_subquery[1].push_back(id);
  }

  /* Handle Joins, Called By DPsize */
  auto handle_join_pairs = [&vertices, &subquery_2_id, &size_subquery](
                               ull left_id, ull right_id, int size) {
    ull subquery = vertices[left_id].subquery_ | vertices[right_id].subquery_;
    ull contribution = vertices[left_id].plan_num_ * vertices[right_id].plan_num_;

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

  /* Subquery with only 1 relation. We must not skip the vertex not contained
     in the sampled subgraph, since they are used later. */
  for (int v = 0; v < vertex_num; ++v) {
    ull id = vertices.size();
    ull subquery = 1ULL << v;
    ull connected = adj_matrix[v];
    vertices.emplace_back(subquery, id, connected);
    vertices.back().plan_num_ = 1;
    subquery_2_id[subquery] = id;
    size_subquery[1].push_back(id);
  }

  /* Handle Joins, Called By DPsize */
  auto handle_join_pairs = [&vertices, &subquery_2_id, &size_subquery](
                               ull left_id, ull right_id, int size) {
    ull subquery = vertices[left_id].subquery_ | vertices[right_id].subquery_;
    ull contribution = vertices[left_id].plan_num_ * vertices[right_id].plan_num_;

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
    int relation = __builtin_ctzll(psg_vertex.subquery_);
    plan_cache[psg_vertex.id_].push_back(std::make_shared<Scan>(relation));
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

    for(const auto &l : left){
      for(const auto &r : right){
        plans.emplace_back(std::make_shared<Join>(l, r));
      }
    }
  }
}

std::vector<std::shared_ptr<PlanNode>> PlanSpaceGraph::sample_plan(
    const PlanSpaceVertex &psg_vertex, ull sample_num,
    std::mt19937_64 &g) const
{
  if (sample_num == 0) return {};

  // Leaf node: a single relation has exactly one join order.
  if (psg_vertex.is_leaf()) {
    int relation = __builtin_ctzll(psg_vertex.subquery_);
    return {std::make_shared<Scan>(relation)};
  }

  // Internal node: Join. Decide how many sampled plans each child pair gets.
  const size_t num_pairs = psg_vertex.child_.size();

  std::vector<ull> left_id(num_pairs), right_id(num_pairs), pair_capacity(num_pairs);
  ull total_capacity = 0;

  for (size_t i = 0; i < num_pairs; ++i) {
    auto [l, r] = psg_vertex.child_[i];
    left_id[i] = l;  right_id[i] = r;
    ull capacity = vertices_[l].plan_num_ * vertices_[r].plan_num_;
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
    ull left_num = vertices_[lid].plan_num_;
    ull right_num = vertices_[rid].plan_num_;

    ull left_sample = std::min(left_num, cur_num);
    ull right_sample = std::min(right_num, cur_num);

    ull minimum_right_plan_required = (cur_num + left_sample - 1) / left_sample;
    right_sample = std::min(right_sample, minimum_right_plan_required);

    auto left_plans = sample_plan(vertices_[lid], left_sample, g);
    auto right_plans = sample_plan(vertices_[rid], right_sample, g);

    ull cnt = 0;
    for(auto &l : left_plans){
      if(cnt == cur_num) break;
      for(auto &r : right_plans){
        result.emplace_back(std::make_shared<Join>(l, r));
        if(++cnt == cur_num) break;
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
  std::mt19937_64 g(42);  // enable reproducibility
  return sample_plan(*root_, limit_num, g);
}
