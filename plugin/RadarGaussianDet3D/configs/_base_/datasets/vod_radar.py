dataset_type = 'VoDDataset'
dataset_root = '/gs/home/zhubing/xwy_project/view_of_delft_PUBLIC/'

input_modality = dict(
    use_lidar=False,
    use_camera=False,
    use_radar=True
)

class_names = ['Car', 'Pedestrian', 'Cyclist']

radar_use_dim = 7
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
post_center_range = [-5, -30.6, -6, 56.2, 30.6, 5]

radar_pts_norm_cfg = dict(
    mean=[0, 0, 0, -12.339, -2.818, 0.037, 0],
    std=[1, 1, 1, 13.240, 1.926, 1.929, 1]
)

train_pipeline = [
    dict(
        type='LoadPointsFromFileV2',
        coord_type='LIDAR',
        sensor='radar',
        load_dim=7,
        use_dim=radar_use_dim,
        **radar_pts_norm_cfg
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='GlobalRotScaleTransV2', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05], translation_std=[0, 0, 0]),
    dict(type='RandomFlip3DV2',
         flip_ratio_bev_horizontal=0.5,
         flip_ratio_bev_vertical=0.0),
    dict(type='PointsRangeFilterV2', point_cloud_range=point_cloud_range, sensor='radar'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffleV2'),
    dict(type='DefaultFormatBundle3DV2'),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'radar_points'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'radar2img', 'cam2img', 'pad_shape',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx',
                    'lidar_filename', 'radar_filename', 'flip', 'pcd_horizontal_flip'))
]
test_pipeline = [
    dict(
        type='LoadPointsFromFileV2',
        coord_type='LIDAR',
        sensor='radar',
        load_dim=7,
        use_dim=radar_use_dim,
        **radar_pts_norm_cfg),
    dict(type='PointsRangeFilterV2', point_cloud_range=point_cloud_range, sensor='radar'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1936, 1216),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3DV2'),
            dict(type='Collect3D', keys=['radar_points'],
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'radar2img', 'cam2img', 'pad_shape',
                            'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx',
                            'lidar_filename', 'radar_filename', 'flip', 'pcd_horizontal_flip'))
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'VoD_infos_train.pkl',
        pts_prefix='velodyne_reduced',
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        box_type_3d='LiDAR',
        pcd_limit_range=post_center_range),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'VoD_infos_val.pkl',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        pcd_limit_range=post_center_range),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'VoD_infos_val.pkl',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        pcd_limit_range=post_center_range))
