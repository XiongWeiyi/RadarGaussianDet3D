#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
    void render(
        const dim3 grid, dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        const float2* means2D,
        const float4* conic_opacity,
        const float* features,
        const float* final_Ts,
        const uint32_t* n_contrib,
        const float* dL_dout_feats,
        float3* dL_dmean2D,
        float4* dL_dconic2D,
        float* dL_dopacity,
        float* dL_dfeatures,
        float* bev_range);

    void preprocess(
        int P,
        const float3* means,
        const int* radii,
        const glm::vec3* scales,
        const glm::vec4* rotations,
        const float scale_modifier,
        const float* cov3Ds,
        const float3* dL_dmean2D,
        const float* dL_dconics,
        const float* bev_range,
        const int W, const int H,
        glm::vec3* dL_dmeans,
        float* dL_dcov3D,
        glm::vec3* dL_dscale,
        glm::vec4* dL_drot);
}

#endif