#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


__device__ float3 computeCov2D(const float* cov3D, const float* bev_range, const int W, const int H)
{
    glm::mat2 J = glm::mat2(
        W / (bev_range[2] - bev_range[0]), 0,
        0, H / (bev_range[3] - bev_range[1])
    );

    glm::mat2 Vrk = glm::mat2(
        cov3D[0], cov3D[1],
        cov3D[1], cov3D[3]
    );

    glm::mat2 cov = J * Vrk * J;

    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    glm::vec4 q = rot;
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;
    glm::mat3 Sigma = glm::transpose(M) * M;

    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

__global__ void preprocessCUDA(int P,
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
    uint32_t* tiles_touched)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    radii[idx] = 0;
    tiles_touched[idx] = 0;

    float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    float2 point_bev = { norm2Pix((p_orig.x - bev_range[0]) / (bev_range[2] - bev_range[0]), W),
                         norm2Pix((p_orig.y - bev_range[1]) / (bev_range[3] - bev_range[1]), H) };

    const float* cov3D;
	if (cov3D_precomp != nullptr)
        cov3D = cov3D_precomp + idx * 6;
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

    float3 cov = computeCov2D(cov3D, bev_range, W, H);

    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f)
        return;
    float det_inv = 1.f / det;
    float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
    uint2 rect_min, rect_max;
    getRect(point_bev, my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    heights[idx] = p_orig.z;
    radii[idx] = my_radius;
    points_xy_bev[idx] = point_bev;
    conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

void FORWARD::preprocess(int P,
    const float* means3D,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* cov3D_precomp,
    const float* bev_range,
    const int W, int H,
    int* radii,
    float2* means2D,
    float* heights,
    float* cov3Ds,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched)
{
    preprocessCUDA << <(P + 255) / 256, 256 >> > (
        P,
        means3D,
        scales,
        scale_modifier,
        rotations,
        opacities,
        cov3D_precomp,
        bev_range,
        W, H,
        radii,
        means2D,
        heights,
        cov3Ds,
        conic_opacity,
        grid,
        tiles_touched);
}

template <uint32_t NUM_CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ points_xy_bev,
    const float* __restrict__ features,
    const float4* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    float* __restrict__ out_feat_map)
{
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = { (float)pix.x, (float)pix.y };

    bool inside = pix.x < W && pix.y < H;
    bool done = !inside;

    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

    float T = 1.0f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_bev[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
        }
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            contributor++;

            float2 xy = collected_xy[j];
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            float4 con_o = collected_conic_opacity[j];
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f)
                continue;

            float alpha = min(0.99f, con_o.w * exp(power));
            if (alpha < 1.0f / 255.0f)
                continue;
            float test_T = T * (1 - alpha);
            if (test_T < 0.0001f)
            {
                done = true;
                continue;
            }

            for (int ch = 0; ch < NUM_CHANNELS; ch++)
                out_feat_map[ch * H * W + pix_id] += features[collected_id[j] * NUM_CHANNELS + ch] * alpha * T;

            T = test_T;

            last_contributor = contributor;
        }
    }

    if (inside)
    {
        final_T[pix_id] = T;
        n_contrib[pix_id] = last_contributor;
    }
}

void FORWARD::render(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* means2D,
    const float* features,
    const float4* conic_opacity,
    float* final_T,
    uint32_t* n_contrib,
    float* out_feat_map)
{
    renderCUDA<FEAT_CHANNELS> << <grid, block >> > (
        ranges,
        point_list,
        W, H,
        means2D,
        features,
        conic_opacity,
        final_T,
        n_contrib,
        out_feat_map);
}