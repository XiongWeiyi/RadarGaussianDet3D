#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
    class Rasterizer
    {
    public:
        static int forward(
            std::function<char* (size_t)> geometryBuffer,
            std::function<char* (size_t)> binningBuffer,
            std::function<char* (size_t)> imageBuffer,
            const int P,
            const int width, int height,
            const float* means3D,
            const float* features,
            const float* opacities,
            const float* scales,
            const float scale_modifier,
            const float* rotations,
            const float* cov3D_precomp,
            const float* bev_range,
            float* out_feat_map,
            int* radii = nullptr,
            bool debug = false);

        static void backward(
            const int P, int R,
            const int width, int height,
            const float* means3D,
            const float* features,
            const float* scales,
            const float scale_modifier,
            const float* rotations,
            const float* cov3D_precomp,
            const int* radii,
            char* geom_buffer,
            char* binning_buffer,
            char* img_buffer,
            const float* dL_dout_feats,
            float* dL_dmean2D,
            float* dL_dconic,
            float* dL_dopacity,
            float* dL_dfeatures,
            float* dL_dmean3D,
            float* dL_dcov3D,
            float* dL_dscale,
            float* dL_drot,
            float* bev_range,
            bool debug);
    };
};

#endif