#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& means3D,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& features,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const int image_height,
    const int image_width,
    const torch::Tensor& bev_range,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& features,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& dL_dout_feats,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const torch::Tensor& bev_range,
    const bool debug);