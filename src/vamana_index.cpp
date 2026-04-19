#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <cstdlib>

// ============================================================================
// Destructor
// ============================================================================

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

// ============================================================================
// Greedy Search
// ============================================================================
// Beam search starting from start_node_. Maintains a candidate set of at most
// L nodes, always expanding the closest unvisited node. Returns when no
// unvisited candidates remain.
//
// Uses std::set<Candidate> as an ordered container — simple, correct, and
// easy for students to understand and modify.

struct CandState {
  float dist;
  uint32_t id;
  bool expanded;
  bool operator<(const CandState &other) const {
    if (dist != other.dist)
      return dist < other.dist;
    return id < other.id;
  }
};

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float *query, uint32_t L) const {
  thread_local std::vector<uint16_t> visited_array;
  thread_local uint16_t visited_gen = 0;

  if (visited_array.size() < npts_) {
    visited_array.assign(npts_, 0);
  }
  visited_gen++;
  if (visited_gen == 0) {
    std::fill(visited_array.begin(), visited_array.end(), 0);
    visited_gen = 1;
  }

  std::vector<CandState> candidates;
  candidates.reserve(L + 1);

  uint32_t dist_cmps = 0;
  float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
  dist_cmps++;
  candidates.push_back({start_dist, start_node_, false});
  visited_array[start_node_] = visited_gen;

  while (true) {
    size_t best_idx = candidates.size();
    for (size_t i = 0; i < candidates.size(); i++) {
      if (!candidates[i].expanded) {
        best_idx = i;
        break;
      }
    }
    if (best_idx == candidates.size())
      break;

    candidates[best_idx].expanded = true;
    uint32_t best_node = candidates[best_idx].id;
    float worst_dist = candidates.back().dist;

    std::vector<uint32_t> neighbors;
    {
      std::lock_guard<std::mutex> lock(locks_[best_node]);
      neighbors = graph_[best_node];
    }

    for (size_t i = 0; i < neighbors.size(); i++) {
      if (i + 1 < neighbors.size()) {
        __builtin_prefetch(get_vector(neighbors[i + 1]), 0, 1);
      }
      uint32_t nbr = neighbors[i];

      if (visited_array[nbr] == visited_gen)
        continue;
      visited_array[nbr] = visited_gen;

      float d = compute_l2sq(query, get_vector(nbr), dim_);
      dist_cmps++;

      if (candidates.size() < L || d < worst_dist) {
        CandState new_cand{d, nbr, false};
        auto it =
            std::lower_bound(candidates.begin(), candidates.end(), new_cand);
        // Lower_bound returns the first element that does not compare less than
        // new_cand. We use id tie-breaking to maintain set-like uniqueness,
        // although ID duplicates shouldn't occur easily since we check
        // visited_array. However, checking equality ensures correctness.
        if (it == candidates.end() || it->id != nbr) {
          candidates.insert(it, new_cand);
          if (candidates.size() > L) {
            candidates.pop_back();
          }
          worst_dist = candidates.back().dist;
        }
      }
    }
  }

  std::vector<Candidate> results;
  results.reserve(candidates.size());
  for (const auto &c : candidates) {
    results.push_back({c.dist, c.id});
  }
  return {results, dist_cmps};
}

// ============================================================================
// Robust Prune (Alpha-RNG Rule)
// ============================================================================
// Given a node and a set of candidates, greedily select neighbors that are
// "diverse" — a candidate c is added only if it's not too close to any
// already-selected neighbor (within a factor of alpha).
//
// Formally: add c if for ALL already-chosen neighbors n:
//     dist(node, c) <= alpha * dist(c, n)
//
// This ensures good graph navigability by keeping some long-range edges
// (alpha > 1 makes it easier for a candidate to survive pruning).

void VamanaIndex::robust_prune(uint32_t node,
                               std::vector<Candidate> &candidates, float alpha,
                               uint32_t R) {
  // Remove self from candidates if present
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [node](const Candidate &c) { return c.second == node; }),
      candidates.end());

  // Sort by distance to node (ascending)
  std::sort(candidates.begin(), candidates.end());

  std::vector<uint32_t> new_neighbors;
  new_neighbors.reserve(R);

  for (size_t i = 0; i < candidates.size(); i++) {
    if (i + 1 < candidates.size()) {
      __builtin_prefetch(get_vector(candidates[i + 1].second), 0, 1);
    }
    const auto &[dist_to_node, cand_id] = candidates[i];

    if (new_neighbors.size() >= R)
      break;

    // Check alpha-RNG condition against all already-selected neighbors
    bool keep = true;
    for (uint32_t selected : new_neighbors) {
      float dist_cand_to_selected =
          compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
      if (dist_to_node > alpha * dist_cand_to_selected) {
        keep = false;
        break;
      }
    }

    if (keep)
      new_neighbors.push_back(cand_id);
  }

  graph_[node] = std::move(new_neighbors);
}

// ============================================================================
// Build
// ============================================================================

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    // --- Load data ---
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dimensions: " << dim_ << std::endl;

    if (L < R) {
        std::cerr << "Warning: L (" << L << ") < R (" << R
                  << "). Setting L = R." << std::endl;
        L = R;
    }

    // --- Initialize empty graph and per-node locks ---
    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    // --- Compute Centroid and Pick Medoid Start Node ---
        std::cout << "  Computing medoid start node..." << std::endl;
        std::vector<float> centroid(dim_, 0.0f);

        // 1. Calculate the centroid (mean vector) in parallel
        #pragma omp parallel
        {
            std::vector<float> local_centroid(dim_, 0.0f);
            #pragma omp for
            for (size_t i = 0; i < npts_; i++) {
                const float* vec = get_vector(i);
                for (size_t d = 0; d < dim_; d++) {
                    local_centroid[d] += vec[d];
                }
            }
            
            #pragma omp critical
            {
                for (size_t d = 0; d < dim_; d++) {
                    centroid[d] += local_centroid[d];
                }
            }
        }

        for (size_t d = 0; d < dim_; d++) {
            centroid[d] /= npts_;
        }

        // 2. Find the point closest to the centroid (the medoid)
        float min_dist = std::numeric_limits<float>::max();
        start_node_ = 0;

        #pragma omp parallel
        {
            float local_min_dist = std::numeric_limits<float>::max();
            uint32_t local_best = 0;

            #pragma omp for
            for (size_t i = 0; i < npts_; i++) {
                float dist = compute_l2sq(centroid.data(), get_vector(i), dim_);
                if (dist < local_min_dist) {
                    local_min_dist = dist;
                    local_best = i;
                }
            }

            #pragma omp critical
            {
                if (local_min_dist < min_dist) {
                    min_dist = local_min_dist;
                    start_node_ = local_best;
                }
            }
        }

        std::cout << "  Start node (medoid): " << start_node_ << std::endl;

        // Re-initialize RNG for the permutation step
        std::mt19937 rng(42);

    // --- Create random insertion order ---
    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    // --- Build graph: parallel insertion with per-node locking ---
    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    std::cout << "Building index (R=" << R << ", L=" << L
              << ", alpha=" << alpha << ", gamma=" << gamma
              << ", gammaR=" << gamma_R << ")..." << std::endl;

    Timer build_timer;

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        // Step 1: Search for this point in the current graph to find candidates
        auto [candidates, _dist_cmps] = greedy_search(get_vector(point), L);

        // Step 2: Prune candidates to get this point's neighbors
        // We don't need to lock graph_[point] here because each point appears
        // exactly once in the permutation — only this thread writes to it now.
        robust_prune(point, candidates, alpha, R);

        // Step 3: Add backward edges from each new neighbor back to this point
        for (uint32_t nbr : graph_[point]) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);

            // Add backward edge
            graph_[nbr].push_back(point);

            // Step 4: If neighbor's degree exceeds gamma*R, prune its neighborhood
            if (graph_[nbr].size() > gamma_R) {
                // Build candidate list from current neighbors of nbr
                std::vector<Candidate> nbr_candidates;
                nbr_candidates.reserve(graph_[nbr].size());
                for (uint32_t nn : graph_[nbr]) {
                    float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                    nbr_candidates.push_back({d, nn});
                }
                robust_prune(nbr, nbr_candidates, alpha, R);
            }
        }

        // Progress reporting (from one thread only)
        if (idx % 10000 == 0) {
            #pragma omp critical
            {
                std::cout << "\r  Inserted " << idx << " / " << npts_
                          << " points" << std::flush;
            }
        }
    }

    double build_time = build_timer.elapsed_seconds();

    // Compute average degree
    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++)
        total_edges += graph_[i].size();
    double avg_degree = (double)total_edges / npts_;

    std::cout << "\n  Build complete in " << build_time << " seconds."
              << std::endl;
    std::cout << "  Average out-degree: " << avg_degree << std::endl;
}

// ============================================================================
// Search
// ============================================================================

SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L) const {
    if (L < K) L = K;

    Timer t;
    auto [candidates, dist_cmps] = greedy_search(query, L);
    double latency = t.elapsed_us();

    // Return top-K results
    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

// ============================================================================
// Save / Load
// ============================================================================
// Binary format:
//   [uint32] npts
//   [uint32] dim
//   [uint32] start_node
//   For each node i in [0, npts):
//     [uint32] degree
//     [uint32 * degree] neighbor IDs

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open file for writing: " + path);

    out.write(reinterpret_cast<const char*>(&npts_), 4);
    out.write(reinterpret_cast<const char*>(&dim_), 4);
    out.write(reinterpret_cast<const char*>(&start_node_), 4);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = graph_[i].size();
        out.write(reinterpret_cast<const char*>(&deg), 4);
        if (deg > 0) {
            out.write(reinterpret_cast<const char*>(graph_[i].data()),
                      deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndex::load(const std::string& index_path,
                       const std::string& data_path) {
    // Load data vectors
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    // Load graph
    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open index file: " + index_path);

    uint32_t file_npts, file_dim;
    in.read(reinterpret_cast<char*>(&file_npts), 4);
    in.read(reinterpret_cast<char*>(&file_dim), 4);
    in.read(reinterpret_cast<char*>(&start_node_), 4);

    if (file_npts != npts_ || file_dim != dim_)
        throw std::runtime_error(
            "Index/data mismatch: index has " + std::to_string(file_npts) +
            "x" + std::to_string(file_dim) + ", data has " +
            std::to_string(npts_) + "x" + std::to_string(dim_));

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), 4);
        graph_[i].resize(deg);
        if (deg > 0) {
            in.read(reinterpret_cast<char*>(graph_[i].data()),
                    deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index loaded: " << npts_ << " points, " << dim_
              << " dims, start=" << start_node_ << std::endl;
}
