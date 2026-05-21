_base_ = [
    '../../../configs/_base_/default_runtime.py',
    './_base_/datasets/vod_radar.py',
    './_base_/schedules/vod_schedule.py',
]

plugin = True
plugin_dir = 'plugin/RadarGaussianDet3D/'

point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
post_center_range = [-5, -30.6, -6, 56.2, 30.6, 5]
voxel_size = [0.16, 0.16, 5]
voxel_shape = [int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0]),
               int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1]),
               int((point_cloud_range[5]-point_cloud_range[2])/voxel_size[2])]

embed_dims = 256

model = dict(
    type='RadarGaussianDet3D',
    pts_encoder=dict(
        type='PointGaussianEncoder',
        in_channels=7,
        out_channels=64,
        output_shape=voxel_shape[:2][::-1],
        pc_range=point_cloud_range,
        gaussian_cfg=dict(
            max_offset=0.0,
            max_scale=1.0,
            pred_opacities=False,
        ),
        lfa_cfg=dict(
            radius=0.32,
            reduce='mean',
        ),
        gfa_cfg=True,
    ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='CenterHeadV2',
        in_channels=embed_dims,
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist'])
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range,
            post_center_range=post_center_range,
            max_num=350,
            score_threshold=0.1,
            out_size_factor=2,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_gaussian=dict(type='KLDivergenceLoss', loss_weight=1.0, scale_factors=[9.0, 1.0, 1.0]),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=voxel_shape,
            voxel_size=voxel_size,
            out_size_factor=2,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=post_center_range,
            min_radius=[4, 0.3, 0.85],
            score_threshold=0.1,
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            grid_size=voxel_shape,
            out_size_factor=2)))
