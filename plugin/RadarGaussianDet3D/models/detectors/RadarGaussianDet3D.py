import torch
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Conv2d
from mmdet3d.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.builder import MODELS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

@DETECTORS.register_module()
class RadarGaussianDet3D(MVXTwoStageDetector):
    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RadarGaussianDet3D, self).__init__(pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, pts_fusion_layer,
                                                 img_backbone, pts_backbone, img_neck, pts_neck, pts_bbox_head,
                                                 img_roi_head, img_rpn_head, train_cfg, test_cfg, pretrained)
        if pts_encoder:
            self.pts_encoder = MODELS.build(pts_encoder)
        if pts_encoder or pts_voxel_encoder:
            self.pts_bev_proj = Conv2d(sum(self.pts_neck.out_channels), self.pts_bbox_head.in_channels, kernel_size=1)

    def init_weights(self):
        pass

    @property
    def with_pts_encoder(self):
        return hasattr(self, 'pts_encoder') and self.pts_encoder is not None

    @property
    def with_voxel_encoder(self):
        return hasattr(self, 'pts_voxel_encoder') and self.pts_voxel_encoder is not None

    def extract_feat(self, lidar_points, radar_points, img, img_metas):
        pts_feats = self.extract_pts_feat(radar_points, img_metas)
        return pts_feats

    @force_fp32()
    def extract_pts_feat(self, pts, img_metas=None):
        x = []
        if self.with_voxel_encoder:
            voxels, num_points, coors = self.voxelize(pts)

            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

            batch_size = coors[-1, 0] + 1
            pts_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)
            x.append(pts_feats)
        if self.with_pts_encoder:
            pts_feats = self.pts_encoder(pts)
            x.append(pts_feats)
        x = torch.cat(x, dim=1)

        x = self.pts_backbone(x)
        x = self.pts_neck(x)[0]
        x = self.pts_bev_proj(x)
        return x

    def forward_pts_train(self,
                          bev_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.pts_bbox_head([bev_feats], img_metas=img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      lidar_points=None,
                      radar_points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        bev_feats = self.extract_feat(lidar_points, radar_points, img, img_metas)

        losses = self.forward_pts_train(bev_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)

        return losses

    def forward_test(self, lidar_points=None, radar_points=None, img_metas=None, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))
        img = [img] if img is None else img
        radar_points = [radar_points] if radar_points is None else radar_points
        lidar_points = [lidar_points] if lidar_points is None else lidar_points
        return self.simple_test(lidar_points[0], radar_points[0], img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False):
        outs = self.pts_bbox_head([x], img_metas=img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, [img_metas[0]], rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
    
    def simple_test(self, lidar_points=None, radar_points=None, img_metas=None, img=None, rescale=False):
        bev_feats = self.extract_feat(lidar_points, radar_points, img, img_metas)

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(bev_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list
