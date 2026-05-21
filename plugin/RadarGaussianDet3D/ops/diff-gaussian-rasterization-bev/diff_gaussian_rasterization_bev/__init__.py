from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        features,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        args = (
            means3D,
            opacities,
            scales,
            rotations,
            features,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.bev_height,
            raster_settings.bev_width,
            raster_settings.bev_range,
            raster_settings.debug
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, out_feat_map, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, out_feat_map, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, features, scales, rotations, cov3Ds_precomp, radii, geomBuffer, binningBuffer, imgBuffer)
        return out_feat_map

    @staticmethod
    def backward(ctx, grad_out_feats):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        means3D, features, scales, rotations, cov3Ds_precomp, radii, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        args = (means3D,
                radii,
                features,
                scales, 
                rotations, 
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                grad_out_feats,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.bev_range,
                raster_settings.debug)

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                grad_means2D, grad_features, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_features, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_features,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    scale_modifier: float
    bev_range: torch.Tensor
    bev_height: int
    bev_width: int
    debug: bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, opacities, features=None, scales=None, rotations=None, cov3D_precomp=None):
        raster_settings = self.raster_settings

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
                (scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])

        return _RasterizeGaussians.apply(
            means3D,
            features,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )
