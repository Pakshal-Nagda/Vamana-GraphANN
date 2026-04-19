#ifndef DISTANCE_H
#define DISTANCE_H

#include <cstdint>

float compute_l2sq(const float* a, const float* b, uint32_t dim);

// NEW: Computes L2 squared up to a specific coordinate limit
float compute_l2sq_approx(const float* a, const float* b, uint32_t limit);

#endif