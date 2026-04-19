#ifndef VAMANA_INDEX_H
#define VAMANA_INDEX_H

#include <vector>
#include <string>
#include <mutex>
#include <memory>

struct SearchResult {
    std::vector<uint32_t> ids;
    uint32_t dist_cmps;
    double latency_us;
};

class VamanaIndex {
public:
    using Candidate = std::pair<float, uint32_t>;

    VamanaIndex() = default;
    ~VamanaIndex();

    void build(const std::string& data_path, uint32_t R, uint32_t L, float alpha, float gamma);
    SearchResult search(const float* query, uint32_t K, uint32_t L, uint32_t k_approx = 0) const;

    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    uint32_t get_dim() const { return dim_; }

private:
    float* data_ = nullptr;
    float* rotated_data_ = nullptr; // NEW: Pre-rotated dataset
    uint32_t npts_ = 0;
    uint32_t dim_ = 0;
    uint32_t start_node_ = 0;
    bool owns_data_ = false;

    std::vector<std::vector<uint32_t>> graph_;
    mutable std::vector<std::mutex> locks_;
    
    // NEW: Rotation members
    std::vector<float> rotation_matrix_;
    void generate_random_rotation();
    void apply_rotation(const float* vec, float* out) const;

    const float* get_vector(uint32_t id) const { return data_ + (size_t)id * dim_; }
    const float* get_rotated_vector(uint32_t id) const { return rotated_data_ + (size_t)id * dim_; }

    std::pair<std::vector<Candidate>, uint32_t> 
    greedy_search(const float* query, uint32_t L, uint32_t k_approx) const;

    void robust_prune(uint32_t node, std::vector<Candidate>& candidates, float alpha, uint32_t R);
};

#endif