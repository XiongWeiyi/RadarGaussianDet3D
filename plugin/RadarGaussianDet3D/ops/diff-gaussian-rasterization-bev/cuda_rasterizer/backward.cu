#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void computeCov2DCUDA(int P,
    const int* radii,
    const float* cov3Ds,
    const float* dL_dconics,
    const float* bev_range,
    const int W,
    const int H,
    float* dL_dcov)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    const float* cov3D = cov3Ds + 6 * idx;

    float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };

    glm::mat2 J = glm::mat2(
        W / (bev_range[2] - bev_range[0]), 0,
        0, H / (bev_range[3] - bev_range[1])
    );

    glm::mat2 Vrk = glm::mat2(
        cov3D[0], cov3D[1],
        cov3D[1], cov3D[3]
    );

    glm::mat2 cov2D = J * Vrk * J;

    float a = cov2D[0][0] += 0.3f;
    float b = cov2D[0][1];
    float c = cov2D[1][1] += 0.3f;

    float denom = a * c - b * b;
    float dL_da = 0, dL_db = 0, dL_dc = 0;
    float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

    if (denom2inv != 0)
    {
        dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
        dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
        dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

        dL_dcov[6 * idx + 0] = J[0][0] * J[0][0] * dL_da;
        dL_dcov[6 * idx + 3] = J[1][1] * J[1][1] * dL_dc;
        dL_dcov[6 * idx + 5] = 0;

        dL_dcov[6 * idx + 1] = J[0][0] * J[1][1] * dL_db;
        dL_dcov[6 * idx + 2] = 0;
        dL_dcov[6 * idx + 4] = 0;
    }
    else
    {
        for (int i = 0; i < 6; i++)
            dL_dcov[6 * idx + i] = 0;
    }
}

__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
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

    glm::mat3 S = glm::mat3(1.0f);

    glm::vec3 s = mod * scale;
    S[0][0] = s.x;
    S[1][1] = s.y;
    S[2][2] = s.z;

    glm::mat3 M = S * R;

    const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

    glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
    glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

    glm::mat3 dL_dSigma = glm::mat3(
        dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
        0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
        0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
    );

    glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

    glm::mat3 Rt = glm::transpose(R);
    glm::mat3 dL_dMt = glm::transpose(dL_dM);

    glm::vec3* dL_dscale = dL_dscales + idx;
    dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
    dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
    dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

    dL_dMt[0] *= s.x;
    dL_dMt[1] *= s.y;
    dL_dMt[2] *= s.z;

    glm::vec4 dL_dq;
    dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
    dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
    dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
    dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

    float4* dL_drot = (float4*)(dL_drots + idx);
    *dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}

__global__ void preprocessCUDA(
    int P,
    const float3* means,
    const int* radii,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float scale_modifier,
    const float3* dL_dmean2D,
    glm::vec3* dL_dmeans,
    float* dL_dcov3D,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    glm::vec3 dL_dmean;
    dL_dmean.x = dL_dmean2D[idx].x;
    dL_dmean.y = dL_dmean2D[idx].y;
    dL_dmean.z = 0;

    dL_dmeans[idx] += dL_dmean;

    if (scales)
        computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

void BACKWARD::preprocess(
    int P,
    const float3* means3D,
    const int* radii,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    const float3* dL_dmean2D,
    const float* dL_dconic,
    const float* bev_range,
    const int W,
    const int H,
    glm::vec3* dL_dmean3D,
    float* dL_dcov3D,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot)
{
    computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
        P,
        radii,
        cov3Ds,
        dL_dconic,
        bev_range,
        W, H,
        dL_dcov3D);

	preprocessCUDA << < (P + 255) / 256, 256 >> > (
        P,
        (float3*)means3D,
        radii,
        (glm::vec3*)scales,
        (glm::vec4*)rotations,
        scale_modifier,
        (float3*)dL_dmean2D,
        (glm::vec3*)dL_dmean3D,
        dL_dcov3D,
        dL_dscale,
        dL_drot);
}

template <uint32_t NUM_CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ points_xy_bev,
    const float4* __restrict__ conic_opacity,
    const float* __restrict__ features,
    const float* __restrict__ final_Ts,
    const uint32_t* __restrict__ n_contrib,
    const float* __restrict__ dL_dout_feats,
    float3* __restrict__ dL_dmean2D,
    float4* __restrict__ dL_dconic2D,
    float* __restrict__ dL_dopacity,
    float* __restrict__ dL_dfeatures,
    float* __restrict__ bev_range)
{
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };

    const bool inside = pix.x < W&& pix.y < H;
    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    bool done = !inside;
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

    const float T_final = inside ? final_Ts[pix_id] : 0;
    float T = T_final;

    uint32_t contributor = toDo;
    const int last_contributor = inside ? n_contrib[pix_id] : 0;

    float accum_rec[NUM_CHANNELS] = { 0 };
    float dL_dFeat[NUM_CHANNELS];
    if (inside)
        for (int i = 0; i < NUM_CHANNELS; i++)
            dL_dFeat[i] = dL_dout_feats[i * H * W + pix_id];

    float last_alpha = 0;
    float last_feature[NUM_CHANNELS] = { 0 };

    const float ddelx_dx = W / (bev_range[2] - bev_range[0]);
    const float ddely_dy = H / (bev_range[3] - bev_range[1]);

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        block.sync();
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            const int coll_id = point_list[range.y - progress - 1];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_bev[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
        }
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            contributor--;
            if (contributor >= last_contributor)
                continue;

            const float2 xy = collected_xy[j];
            const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            const float4 con_o = collected_conic_opacity[j];
            const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f)
                continue;

            const float G = exp(power);
            const float alpha = min(0.99f, con_o.w * G);
            if (alpha < 1.0f / 255.0f)
                continue;

            T = T / (1.f - alpha);
            const float dchannel_dfeatures = alpha * T;

            float dL_dalpha = 0.0f;
            const int global_id = collected_id[j];
            for (int ch = 0; ch < NUM_CHANNELS; ch++)
            {
                const float f = features[collected_id[j] * NUM_CHANNELS + ch];
                accum_rec[ch] = last_alpha * last_feature[ch] + (1.f - last_alpha) * accum_rec[ch];
                last_feature[ch] = f;

                const float dL_dchannel = dL_dFeat[ch];
                dL_dalpha += (f - accum_rec[ch]) * dL_dchannel;
                atomicAdd(&(dL_dfeatures[global_id * NUM_CHANNELS + ch]), dchannel_dfeatures * dL_dchannel);
            }
            dL_dalpha *= T;
            last_alpha = alpha;

            const float dL_dG = con_o.w * dL_dalpha;
            const float gdx = G * d.x;
            const float gdy = G * d.y;
            const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
            const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

            atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
            atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

            atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
            atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
            atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

            atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
        }
    }
}

void BACKWARD::render(
    const dim3 grid, const dim3 block,
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
    float* bev_range)
{
    renderCUDA<FEAT_CHANNELS> << <grid, block >> >(
        ranges,
        point_list,
        W, H,
        means2D,
        conic_opacity,
        features,
        final_Ts,
        n_contrib,
        dL_dout_feats,
        dL_dmean2D,
        dL_dconic2D,
        dL_dopacity,
        dL_dfeatures,
        bev_range);
}
