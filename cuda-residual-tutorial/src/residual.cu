#include "cuda_residual_tutorial/residual.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace cuda_residual_tutorial
{
namespace
{
void check_cuda(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

// One thread computes all four residual components for one correspondence.
__global__ void residual_kernel(const Pair* pairs, std::size_t count, Pose pose, Residual* residuals)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;  // The last block may contain more threads than remaining pairs.
    }

    const Pair pair = pairs[i];
    Residual result{};
    for (int row = 0; row < 3; ++row)
    {
        result.point[row] = pose.rotation[3 * row] * pair.source[0] + pose.rotation[3 * row + 1] * pair.source[1] +
                            pose.rotation[3 * row + 2] * pair.source[2] + pose.translation[row] - pair.target[row];
        result.plane += pair.normal[row] * result.point[row];
    }
    residuals[i] = result;
}

// Free both allocations even if the second allocation or a CUDA operation fails.
struct DeviceBuffers
{
    Pair* pairs = nullptr;
    Residual* residuals = nullptr;
    ~DeviceBuffers()
    {
        cudaFree(residuals);
        cudaFree(pairs);
    }
};
}  // namespace

std::vector<Residual> evaluate_cuda(const std::vector<Pair>& pairs, const Pose& pose)
{
    if (pairs.empty())
    {
        return {};
    }
    constexpr unsigned threads = 256;
    const std::size_t blocks = 1 + (pairs.size() - 1) / threads;
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "get CUDA device");
    check_cuda(cudaGetDeviceProperties(&properties, device), "get CUDA device properties");
    if (blocks > static_cast<std::size_t>(properties.maxGridSize[0]) ||
        pairs.size() > std::numeric_limits<std::size_t>::max() / std::max(sizeof(Pair), sizeof(Residual)))
    {
        throw std::runtime_error("too many pairs for a single tutorial kernel launch");
    }

    std::vector<Residual> residuals(pairs.size());
    DeviceBuffers device_buffers;
    // 1. Allocate GPU memory; CPU vectors cannot be dereferenced by the kernel.
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_buffers.pairs), pairs.size() * sizeof(Pair)), "allocate pairs");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_buffers.residuals), residuals.size() * sizeof(Residual)), "allocate residuals");
    // 2. Transfer CPU correspondences to GPU. Pose is passed by value at launch.
    check_cuda(cudaMemcpy(device_buffers.pairs, pairs.data(), pairs.size() * sizeof(Pair), cudaMemcpyHostToDevice), "upload pairs");
    // 3. Evaluate residuals only: no Jacobian, reduction, or pose optimization.
    residual_kernel<<<static_cast<unsigned>(blocks), threads>>>(device_buffers.pairs, pairs.size(), pose, device_buffers.residuals);
    check_cuda(cudaGetLastError(), "launch residual kernel");
    check_cuda(cudaDeviceSynchronize(), "execute residual kernel");
    // 4. Copy results back, retaining the correspondence order.
    check_cuda(cudaMemcpy(residuals.data(), device_buffers.residuals, residuals.size() * sizeof(Residual), cudaMemcpyDeviceToHost), "download residuals");
    return residuals;
}
}  // namespace cuda_residual_tutorial
