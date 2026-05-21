#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

__forceinline__ __device__ float norm2Pix(float v, int S)
{
    return v * S - 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
    rect_min = {
        min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
    };
    rect_max = {
        min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
    };
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
    };
    return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
    float4 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
    };
    return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
    };
    return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
        matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
    };
    return transformed;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif