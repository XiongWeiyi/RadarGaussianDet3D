#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
    void preprocess(int P,
        const float* orig_points,
        const glm::vec3* scales,
        const float scale_modifier,
        const glm::vec4* rotations,
        const float* opacities,
        const float* cov3D_precomp,
        const float* bev_range,
        const int W, int H,
        int* radii,
        float2* points_xy_bev,
        float* heights,
        float* cov3Ds,
        float4* conic_opacity,
        const dim3 grid,
        uint32_t* tiles_touched);

    void render(
        const dim3 grid, dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        const float2* points_xy_bev,
        const float* features,
        const float4* conic_opacity,
        float* final_T,
        uint32_t* n_contrib,
        float* out_feat_map);
}


#endif