#pragma once

#include <ceres/evaluation_callback.h>
#include <ceres/problem.h>
#include <ceres/sized_cost_function.h>
#include <Eigen/Core>
#include <cstddef>
#include <memory>
#include <string>

namespace p2ptpl_icp
{
// Implemented by the library, so downstream defaults match its build configuration.
bool CudaEvaluationCompiled() noexcept;

#ifdef P2PTPL_HAS_CUDA
// One batch for all correspondences. Reset retains CUDA allocations between outer
// ICP iterations. The callback and xi must outlive their Ceres problem. Add/Reset
// must never run while Ceres is evaluating the problem.
class CudaErrorEvaluation : public ceres::EvaluationCallback
{
public:
    CudaErrorEvaluation(double point_weight, double plane_weight);
    ~CudaErrorEvaluation() override;
    CudaErrorEvaluation(const CudaErrorEvaluation&) = delete;
    CudaErrorEvaluation& operator=(const CudaErrorEvaluation&) = delete;

    void Reset(const double* xi);
    std::size_t Add(const Eigen::Vector3d& source, const Eigen::Vector3d& target, const Eigen::Vector3d& normal);
    void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) override;
    bool Copy(std::size_t index, bool plane, double* residuals, double* jacobian) const;
    const std::string& last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

template <bool Plane>
class CudaCostFunction : public ceres::SizedCostFunction<Plane ? 1 : 3, 6>
{
public:
    CudaCostFunction(const CudaErrorEvaluation& cache, std::size_t index) : cache_(cache), index_(index)
    {
    }

    bool Evaluate(double const* const*, double* residuals, double** jacobians) const override
    {
        return cache_.Copy(index_, Plane, residuals, jacobians ? jacobians[0] : nullptr);
    }

private:
    const CudaErrorEvaluation& cache_;
    std::size_t index_;
};

inline void AddCudaCorrespondenceResiduals(ceres::Problem& problem, const CudaErrorEvaluation& cache, std::size_t index, double* xi, ceres::LossFunction* loss)
{
    // Separate blocks preserve the original per-point and per-plane robust losses.
    problem.AddResidualBlock(new CudaCostFunction<false>(cache, index), loss, xi);
    problem.AddResidualBlock(new CudaCostFunction<true>(cache, index), loss, xi);
}
#endif
}  // namespace p2ptpl_icp
