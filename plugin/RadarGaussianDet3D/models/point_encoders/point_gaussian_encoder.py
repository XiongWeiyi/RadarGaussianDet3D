import torch
from mmcv.runner import force_fp32
from torch import nn
from mmdet3d.models.builder import MODELS

@MODELS.register_module()
class PointGaussianEncoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=64, output_shape=None, pc_range=None,
                 lfa_cfg=None, gfa_cfg=None, gaussian_cfg=dict(), **kwargs):
        super(PointGaussianEncoder, self).__init__()
        self.device = kwargs.get('device', 'cuda')
        self.pc_range = pc_range
        self.bev_range = torch.tensor([pc_range[0], pc_range[1], pc_range[3], pc_range[4]], device=self.device)
        self.output_shape = output_shape

        in_channels_ = in_channels

        self.lfa = True if lfa_cfg is not None else False
        if self.lfa:
            self.radius = lfa_cfg.get('radius', 0.32)
            self.reduce = lfa_cfg.get('reduce', 'mean')
            self.pts_proj = nn.Linear(in_channels_ + 3, out_channels, device=self.device)

        self.gfa = True if gfa_cfg is not None else False
        if self.gfa:
            self.pts_proj2 = nn.Linear(in_channels_, out_channels, device=self.device)
            self.global_feature_aggragate = PointAttention(num_channels=out_channels)

        self.max_offset = gaussian_cfg.get('max_offset', 0.0)
        self.max_scale = gaussian_cfg.get('max_scale', 1.0)
        self.pred_opacities = gaussian_cfg.get('pred_opacities', False)

        in_channels += 3
        if self.lfa:
            in_channels += out_channels
        if self.gfa:
            in_channels += out_channels
        self.gaussian_net = nn.Linear(in_channels, out_channels + 11, device=self.device)
        nn.init.zeros_(self.gaussian_net.weight[:3])
        nn.init.zeros_(self.gaussian_net.bias[:3])

        from diff_gaussian_rasterization_bev import GaussianRasterizationSettings, GaussianRasterizer
        bev_raster_settings = GaussianRasterizationSettings(scale_modifier=1, bev_range=self.bev_range,
                                                            bev_height=self.output_shape[0], bev_width=self.output_shape[1],
                                                            debug=False)
        self.bev_rasterizer = GaussianRasterizer(bev_raster_settings)

    @force_fp32()
    def render(self, gaussians, pos):
        means3D = (gaussians[..., 0:3].sigmoid() * 2 - 1) * self.max_offset + pos
        scales = gaussians[..., 3:6].sigmoid() * self.max_scale
        rotations = torch.nn.functional.normalize(gaussians[..., 6:10], p=2, dim=-1)
        if self.pred_opacities:
            opacities = torch.sigmoid(gaussians[..., 10:11])
        else:
            opacities = torch.ones_like(gaussians[..., 10:11])

        features = gaussians[..., 11:]

        featmap = self.bev_rasterizer(means3D, opacities, features, scales, rotations)
        return featmap

    def forward(self, points):
        bev_feats = []

        for frame_id in range(len(points)):
            pts_feats = []

            pos = points[frame_id][:, :3]
            ranges = torch.norm(pos, dim=-1, keepdim=True, p=2)
            azimuths = torch.atan2(pos[:, 1:2], pos[:, 0:1])
            elevations = torch.acos(pos[:, 2:3] / ranges)

            points_frame = torch.cat([points[frame_id]], dim=-1)

            if self.lfa:
                pts_feats.append(self.local_feature_aggragate(points_frame))
            if self.gfa:
                pts_feats.append(self.global_feature_aggragate(self.pts_proj2(points_frame)))

            pts = torch.cat([ranges, azimuths, elevations, points[frame_id], *pts_feats], dim=-1)
            gaussians = self.gaussian_net(pts)
            featmap = self.render(gaussians, pos=pos)
            bev_feats.append(featmap)

        bev_feats = torch.stack(bev_feats, dim=0)
        return bev_feats


    def local_feature_aggragate(self, points):
        dist = (points[:, :3].unsqueeze(0) - points[:, :3].unsqueeze(1)).norm(p=2, dim=-1)
        mask = (dist < self.radius)

        indices = mask.nonzero(as_tuple=True)
        neighbor_pts = points[indices[1]]
        center_pts = points[indices[0]][:, :3]
        pts_feats = torch.cat([neighbor_pts, neighbor_pts[:, :3] - center_pts], dim=-1)

        pts_feats = self.pts_proj(pts_feats)

        new_pts_feats = torch.zeros((len(points), pts_feats.shape[-1]), device=self.device)
        new_pts_feats.index_reduce_(dim=0, index=indices[0], source=pts_feats, reduce=self.reduce, include_self=False)

        return new_pts_feats


class PointAttention(nn.Module):
    def __init__(self, num_channels):
        super(PointAttention, self).__init__()
        # MLP layers for generating Q, K, V
        self.q_mlp = nn.Linear(num_channels, num_channels)
        self.k_mlp = nn.Linear(num_channels, num_channels)
        self.v_mlp = nn.Linear(num_channels, num_channels)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)

    def forward(self, point_features):
        # Gather non-empty pillars based on sparsity mask
        point_features = self.norm1(point_features)
        # Linear projections to get Q, K, V
        Q = self.q_mlp(point_features)  # (p, E)
        K = self.k_mlp(point_features)  # (p, E)
        V = self.v_mlp(point_features)  # (p, E)

        # Compute attention scores
        attention_scores = Q @ K.transpose(-2, -1)  # (p, p)
        attention_weights = attention_scores.softmax(dim=-1)

        # Compute attended features
        attended_features = attention_weights @ V  # (p, E)
        attended_features = attended_features + point_features

        # Pass through FFN
        updated_points = self.ffn(self.norm2(attended_features))  # (p, C)

        updated_points = updated_points + attended_features
        return updated_points