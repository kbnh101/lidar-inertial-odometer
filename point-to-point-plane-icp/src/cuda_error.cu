#include "p2ptpl_icp/cuda_error.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "common/rotation.hpp"

namespace p2ptpl_icp
{
namespace
{
void CheckCuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct Pair
{
    double source[3], target[3], normal[3];
};

struct Pose
{
    double translation[3];
    double rotation[9];
    double derivatives[27];  // alpha, beta, gamma, each row major
    double point_scale, plane_scale;
};

// Each thread evaluates one complete 3D + 1D pair in double precision. Ceres
// expects row-major Jacobian blocks, hence the packed 18 + 6 output layout.
template <bool Jacobians>
__global__ void EvaluatePairs(const Pair* pairs, std::size_t count, Pose pose, double* residuals, double* jacobians)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const Pair pair = pairs[index];
    double error[3];
    for (int row = 0; row < 3; ++row)
    {
        error[row] = pose.rotation[3 * row] * pair.source[0] + pose.rotation[3 * row + 1] * pair.source[1] + pose.rotation[3 * row + 2] * pair.source[2] +
                     pose.translation[row] - pair.target[row];
        residuals[4 * index + row] = pose.point_scale * error[row];
    }
    residuals[4 * index + 3] = pose.plane_scale * (pair.normal[0] * error[0] + pair.normal[1] * error[1] + pair.normal[2] * error[2]);

    if (Jacobians)
    {
        for (int col = 0; col < 6; ++col)
        {
            double plane_jacobian = 0.0;
            for (int row = 0; row < 3; ++row)
            {
                double value = row == col ? 1.0 : 0.0;
                if (col >= 3)
                {
                    const double* derivative = pose.derivatives + 9 * (col - 3) + 3 * row;
                    value = derivative[0] * pair.source[0] + derivative[1] * pair.source[1] + derivative[2] * pair.source[2];
                }
                jacobians[24 * index + 6 * row + col] = pose.point_scale * value;
                plane_jacobian += pair.normal[row] * value;
            }
            jacobians[24 * index + 18 + col] = pose.plane_scale * plane_jacobian;
        }
    }
}

// RAII covers partial allocation failure as well as the normal destruction path.
struct Buffers
{
    Pair* pairs = nullptr;
    double* residuals = nullptr;
    double* jacobians = nullptr;
    double* host_residuals = nullptr;
    double* host_jacobians = nullptr;
    std::size_t capacity = 0;

    ~Buffers()
    {
        cudaFree(pairs);
        cudaFree(residuals);
        cudaFree(jacobians);
        cudaFreeHost(host_residuals);
        cudaFreeHost(host_jacobians);
    }

    void Allocate(std::size_t count)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&pairs), count * sizeof(Pair)), "allocate CUDA correspondences");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&residuals), count * 4 * sizeof(double)), "allocate CUDA residuals");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&jacobians), count * 24 * sizeof(double)), "allocate CUDA Jacobians");
        CheckCuda(cudaMallocHost(reinterpret_cast<void**>(&host_residuals), count * 4 * sizeof(double)), "allocate pinned residuals");
        CheckCuda(cudaMallocHost(reinterpret_cast<void**>(&host_jacobians), count * 24 * sizeof(double)), "allocate pinned Jacobians");
        capacity = count;
    }
};
}  // namespace

struct CudaErrorEvaluation::Impl
{
    cudaStream_t stream = nullptr;
    std::unique_ptr<Buffers> buffers;
    std::vector<Pair> pairs;
    const double* xi = nullptr;
    double point_scale, plane_scale;
    bool uploaded = false;
    bool residuals_ready = false;
    bool jacobians_ready = false;
    std::string error;

    Impl(double point_weight, double plane_weight) : point_scale(std::sqrt(point_weight)), plane_scale(std::sqrt(plane_weight))
    {
        if (!std::isfinite(point_weight) || !std::isfinite(plane_weight) || point_weight <= 0 || plane_weight <= 0)
            throw std::invalid_argument("both CUDA residual weights must be finite and positive");
        CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "initialize CUDA residual evaluator");
    }

    ~Impl()
    {
        cudaStreamSynchronize(stream);
        buffers.reset();
        cudaStreamDestroy(stream);
    }

    void Evaluate(bool evaluate_jacobians)
    {
        if (!xi)
            throw std::logic_error("call CudaErrorEvaluation::Reset with the pose before evaluation");
        if (pairs.empty())
            return;
        if (!buffers || buffers->capacity < pairs.size())
        {
            auto next = std::make_unique<Buffers>();
            next->Allocate(std::max(pairs.size(), buffers ? buffers->capacity * 2 : std::size_t(256)));
            buffers = std::move(next);
        }
        if (!uploaded)
        {
            // Correspondences are fixed for this Ceres solve; upload only once.
            CheckCuda(cudaMemcpyAsync(buffers->pairs, pairs.data(), pairs.size() * sizeof(Pair), cudaMemcpyHostToDevice, stream),
                      "upload CUDA correspondences");
            uploaded = true;
        }

        // Pose-wide coefficients are computed once on the host, as in the CPU
        // callback. All per-correspondence residual/Jacobian arithmetic runs on GPU.
        Pose pose{};
        std::copy_n(xi, 3, pose.translation);
        using RowMatrix = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
        Eigen::Map<RowMatrix> rotation(pose.rotation);
        rotation = common::euler_zyx_to_rotation(xi[3], xi[4], xi[5]);
        if (evaluate_jacobians)
        {
            Eigen::Map<RowMatrix> da(pose.derivatives), db(pose.derivatives + 9), dg(pose.derivatives + 18);
            da = common::rotation_derivative_alpha(xi[3], xi[4], xi[5]);
            db = common::rotation_derivative_beta(xi[3], xi[4], xi[5]);
            dg = common::rotation_derivative_gamma(xi[3], xi[4], xi[5]);
        }
        pose.point_scale = point_scale;
        pose.plane_scale = plane_scale;
        const unsigned blocks = static_cast<unsigned>((pairs.size() + 127) / 128);
        if (evaluate_jacobians)
            EvaluatePairs<true><<<blocks, 128, 0, stream>>>(buffers->pairs, pairs.size(), pose, buffers->residuals, buffers->jacobians);
        else
            EvaluatePairs<false><<<blocks, 128, 0, stream>>>(buffers->pairs, pairs.size(), pose, buffers->residuals, buffers->jacobians);
        CheckCuda(cudaGetLastError(), "launch CUDA residual/Jacobian kernel");
        CheckCuda(cudaMemcpyAsync(buffers->host_residuals, buffers->residuals, pairs.size() * 4 * sizeof(double), cudaMemcpyDeviceToHost, stream),
                  "download CUDA residuals");
        if (evaluate_jacobians)
            CheckCuda(cudaMemcpyAsync(buffers->host_jacobians, buffers->jacobians, pairs.size() * 24 * sizeof(double), cudaMemcpyDeviceToHost, stream),
                      "download CUDA Jacobians");
        // Ceres can read the pinned cache concurrently only after this barrier.
        CheckCuda(cudaStreamSynchronize(stream), "finish CUDA residual/Jacobian evaluation");
    }
};

CudaErrorEvaluation::CudaErrorEvaluation(double point_weight, double plane_weight) : impl_(std::make_unique<Impl>(point_weight, plane_weight))
{
}
CudaErrorEvaluation::~CudaErrorEvaluation() = default;

void CudaErrorEvaluation::Reset(const double* xi)
{
    impl_->xi = xi;
    impl_->pairs.clear();
    impl_->uploaded = impl_->residuals_ready = impl_->jacobians_ready = false;
    impl_->error.clear();
}

std::size_t CudaErrorEvaluation::Add(const Eigen::Vector3d& source, const Eigen::Vector3d& target, const Eigen::Vector3d& normal)
{
    Pair pair;
    std::copy_n(source.data(), 3, pair.source);
    std::copy_n(target.data(), 3, pair.target);
    std::copy_n(normal.data(), 3, pair.normal);
    impl_->pairs.push_back(pair);
    impl_->uploaded = impl_->residuals_ready = impl_->jacobians_ready = false;
    return impl_->pairs.size() - 1;
}

void CudaErrorEvaluation::PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point)
{
    if (!impl_->error.empty())
        return;  // CUDA failures fail the solve; never silently switch to CPU.
    if (new_evaluation_point)
        impl_->residuals_ready = impl_->jacobians_ready = false;
    if (impl_->residuals_ready && (!evaluate_jacobians || impl_->jacobians_ready))
        return;
    try
    {
        impl_->Evaluate(evaluate_jacobians);
        impl_->residuals_ready = true;
        impl_->jacobians_ready = evaluate_jacobians;
    }
    catch (const std::exception& error)
    {
        // EvaluationCallback returns void. CostFunction::Evaluate reports false
        // to Ceres, and the caller can retrieve the CUDA diagnostic afterwards.
        cudaStreamSynchronize(impl_->stream);
        impl_->error = error.what();
        impl_->residuals_ready = impl_->jacobians_ready = false;
    }
}

bool CudaErrorEvaluation::Copy(std::size_t index, bool plane, double* residuals, double* jacobian) const
{
    if (index >= impl_->pairs.size() || !impl_->residuals_ready || (jacobian && !impl_->jacobians_ready))
        return false;
    const int rows = plane ? 1 : 3;
    const double* cached = impl_->buffers->host_residuals + 4 * index + (plane ? 3 : 0);
    std::memcpy(residuals, cached, rows * sizeof(double));
    if (jacobian)
        std::memcpy(jacobian, impl_->buffers->host_jacobians + 24 * index + (plane ? 18 : 0), rows * 6 * sizeof(double));
    return std::all_of(residuals, residuals + rows,
                       [](double value)
                       {
                           return std::isfinite(value);
                       });
}

const std::string& CudaErrorEvaluation::last_error() const
{
    return impl_->error;
}
}  // namespace p2ptpl_icp
