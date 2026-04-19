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
#include <cmath>
#include <limits>

// ============================================================================
// Destructor
// ============================================================================

VamanaIndex::~VamanaIndex() {
    if (owns_data_) {
        if (data_) std::free(data_);
        if (rotated_data_) std::free(rotated_data_);
    }
}

// ============================================================================
// Rotation & Medoid Utilities
// ============================================================================

void VamanaIndex::generate_random_rotation() {
    rotation_matrix_.assign((size_t)dim_ * dim_, 0.0f);
    std::mt19937 gen(42); 
    std::normal_distribution<float> dist(0.0, 1.0);

    for (auto& val : rotation_matrix_) val = dist(gen);

    // Gram-Schmidt to produce orthonormal basis
    for (uint32_t i = 0; i < dim_; ++i) {
        float* row_i = &rotation_matrix_[i * dim_];
        for (uint32_t j = 0; j < i; ++j) {
            float* row_j = &rotation_matrix_[j * dim_];
            float dot = 0;
            for (uint32_t d = 0; d < dim_; ++d) dot += row_i[d] * row_j[d];
            for (uint32_t d = 0; d < dim_; ++d) row_i[d] -= dot * row_j[d];
        }
        float norm = 0;
        for (uint32_t d = 0; d < dim_; ++d) norm += row_i[d] * row_i[d];
        norm = std::sqrt(norm);
        if (norm > 1e-9f) {
            for (uint32_t d = 0; d < dim_; ++d) row_i[d] /= norm;
        }
    }
}

void VamanaIndex::apply_rotation(const float* vec, float* out) const {
    for (uint32_t i = 0; i < dim_; ++i) {
        float sum = 0;
        const float* rot_row = &rotation_matrix_[i * dim_];
        for (uint32_t j = 0; j < dim_; ++j) {
            sum += rot_row[j] * vec[j];
        }
        out[i] = sum;
    }
}

// ============================================================================
// Greedy Search (Merged with Approximation Support)
// ============================================================================

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L, uint32_t k_approx) const {
    std::set<Candidate> candidate_set;
    std::vector<bool> visited(npts_, false);
    uint32_t dist_cmps = 0;
    
    uint32_t effective_k = (k_approx > 0 && k_approx < dim_) ? k_approx : dim_;

    // Rotate query for approximate distance comparison
    std::vector<float> rotated_query(dim_);
    apply_rotation(query, rotated_query.data());

    // Seed with start node
    float start_dist = compute_l2sq_approx(rotated_query.data(), get_rotated_vector(start_node_), effective_k);
    dist_cmps++;
    candidate_set.insert({start_dist, start_node_});
    visited[start_node_] = true;

    std::set<uint32_t> expanded;

    while (true) {
        uint32_t best_node = UINT32_MAX;
        for (const auto& [dist, id] : candidate_set) {
            if (expanded.find(id) == expanded.end()) {
                best_node = id;
                break;
            }
        }
        if (best_node == UINT32_MAX) break;

        expanded.insert(best_node);

        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }

        for (uint32_t nbr : neighbors) {
            if (visited[nbr]) continue;
            visited[nbr] = true;

            float d = compute_l2sq_approx(rotated_query.data(), get_rotated_vector(nbr), effective_k);
            dist_cmps++;

            if (candidate_set.size() < L) {
                candidate_set.insert({d, nbr});
            } else {
                auto worst = std::prev(candidate_set.end());
                if (d < worst->first) {
                    candidate_set.erase(worst);
                    candidate_set.insert({d, nbr});
                }
            }
        }
    }

    return {{candidate_set.begin(), candidate_set.end()}, dist_cmps};
}

// ============================================================================
// Robust Prune & Build (Merged with Medoid Logic)
// ============================================================================

void VamanaIndex::robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());

    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R) break;

        bool keep = true;
        for (uint32_t selected : new_neighbors) {
            float dist_cand_to_selected = compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
            if (dist_to_node > alpha * dist_cand_to_selected) {
                keep = false;
                break;
            }
        }
        if (keep) new_neighbors.push_back(cand_id);
    }
    graph_[node] = std::move(new_neighbors);
}

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    // 1. Generate rotation and pre-rotate dataset
    generate_random_rotation();
    size_t data_size = (size_t)npts_ * dim_ * sizeof(float);
    rotated_data_ = static_cast<float*>(std::aligned_alloc(64, (data_size + 63) & ~(size_t)63));
    
    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < npts_; ++i) {
        apply_rotation(get_vector(i), rotated_data_ + (size_t)i * dim_);
    }

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    // 2. Compute Centroid and Pick Medoid Start Node
    std::cout << "  Computing medoid start node..." << std::endl;
    std::vector<float> centroid(dim_, 0.0f);

    #pragma omp parallel
    {
        std::vector<float> local_centroid(dim_, 0.0f);
        #pragma omp for
        for (size_t i = 0; i < npts_; i++) {
            const float* vec = get_vector(i);
            for (size_t d = 0; d < dim_; d++) local_centroid[d] += vec[d];
        }
        #pragma omp critical
        {
            for (size_t d = 0; d < dim_; d++) centroid[d] += local_centroid[d];
        }
    }
    for (size_t d = 0; d < dim_; d++) centroid[d] /= npts_;

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

    // 3. Build Graph
    std::mt19937 rng(42);
    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    Timer build_timer;

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];
        // Build phase always uses full distance (k_approx = 0)
        auto [candidates, _dist_cmps] = greedy_search(get_vector(point), L, 0);
        robust_prune(point, candidates, alpha, R);

        for (uint32_t nbr : graph_[point]) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);
            graph_[nbr].push_back(point);

            if (graph_[nbr].size() > gamma_R) {
                std::vector<Candidate> nbr_candidates;
                for (uint32_t nn : graph_[nbr]) {
                    float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                    nbr_candidates.push_back({d, nn});
                }
                robust_prune(nbr, nbr_candidates, alpha, R);
            }
        }
    }
    std::cout << "Build complete in " << build_timer.elapsed_seconds() << " seconds." << std::endl;
}

// ============================================================================
// Search & Persistence (Merged)
// ============================================================================

SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L, uint32_t k_approx) const {
    if (L < K) L = K;
    Timer t;
    auto [candidates, dist_cmps] = greedy_search(query, L, k_approx);
    double latency = t.elapsed_us();

    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Cannot open file: " + path);

    out.write(reinterpret_cast<const char*>(&npts_), 4);
    out.write(reinterpret_cast<const char*>(&dim_), 4);
    out.write(reinterpret_cast<const char*>(&start_node_), 4);

    out.write(reinterpret_cast<const char*>(rotation_matrix_.data()), rotation_matrix_.size() * sizeof(float));

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = graph_[i].size();
        out.write(reinterpret_cast<const char*>(&deg), 4);
        if (deg > 0) out.write(reinterpret_cast<const char*>(graph_[i].data()), deg * sizeof(uint32_t));
    }
}

void VamanaIndex::load(const std::string& index_path, const std::string& data_path) {
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::ifstream in(index_path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&npts_), 4);
    in.read(reinterpret_cast<char*>(&dim_), 4);
    in.read(reinterpret_cast<char*>(&start_node_), 4);

    rotation_matrix_.resize((size_t)dim_ * dim_);
    in.read(reinterpret_cast<char*>(rotation_matrix_.data()), rotation_matrix_.size() * sizeof(float));

    size_t data_size = (size_t)npts_ * dim_ * sizeof(float);
    rotated_data_ = static_cast<float*>(std::aligned_alloc(64, (data_size + 63) & ~(size_t)63));
    for (uint32_t i = 0; i < npts_; ++i) apply_rotation(get_vector(i), rotated_data_ + (size_t)i * dim_);

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);
    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), 4);
        graph_[i].resize(deg);
        if (deg > 0) in.read(reinterpret_cast<char*>(graph_[i].data()), deg * sizeof(uint32_t));
    }
}