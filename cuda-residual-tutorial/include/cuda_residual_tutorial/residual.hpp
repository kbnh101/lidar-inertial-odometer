#pragma once

#include <vector>

namespace cuda_residual_tutorial
{
// Plain doubles keep the CPU/GPU boundary independent of Eigen and Ceres.
struct Pair
{
    double source[3];
    double target[3];
    double normal[3];  // Unit target normal, already normalized by common's loader.
};

struct Pose
{
    double rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // Row-major R.
    double translation[3] = {0, 0, 0};
};

struct Residual
{
    double point[3];  // R * source + t - target (signed vector, not its norm).
    double plane;  // dot(target normal, point).
};

// Synchronous tutorial path: allocate -> upload -> kernel -> download -> free.
// Inputs must be finite. Empty input returns empty output without launching CUDA.
// Output order matches input pair order. CUDA errors throw std::runtime_error.
std::vector<Residual> evaluate_cuda(const std::vector<Pair>& pairs, const Pose& pose);
}  // namespace cuda_residual_tutorial
