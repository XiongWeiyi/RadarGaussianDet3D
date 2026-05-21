from pathlib import Path

import mmcv
import numpy as np

from mmdet3d.core.bbox import box_np_ops
from tools.data_converter.TJ4DRadSet_data_utils import get_TJ4DRadSet_image_info

TJ4DRadSet_categories = ('Car', 'Pedestrian', 'Cyclist', 'Truck', 'Other')

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def _calculate_num_points_in_gt(infos, remove_outside=True):
    for info in mmcv.track_iter_progress(infos):
        radar_info = info['radar_point_cloud']
        image_info = info['image']
        calib = info['calib']

        radar_path = radar_info['velodyne_path']
        radar_features = radar_info['num_features']
        points_r = np.fromfile(radar_path, dtype=np.float32, count=-1).reshape([-1, radar_features])

        rect = calib['R0_rect']
        Trr2c = calib['Tr_radar_to_cam']
        P0 = calib['P0']
        if remove_outside:
            points_r = box_np_ops.remove_outside_points(points_r, rect, Trr2c, P0, image_info['image_shape'])

        annos = info['annos']
        dims = annos['dimensions']
        loc = annos['location']
        rots = annos['rotation_y']
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        gt_boxes_radar = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, Trr2c)
        radar_indices = box_np_ops.points_in_rbbox(points_r[:, :3], gt_boxes_radar)

        num_radar_points_in_gt = radar_indices.sum(0)

        mask = annos['name'] == 'DontCare'
        num_radar_points_in_gt[mask] = -1
        annos['num_radar_points_in_gt'] = num_radar_points_in_gt.astype(np.int32)


def create_TJ4DRadSet_info_file(data_path):
    """Create info file of TJ4DRadSet dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    save_path = Path(data_path)

    TJ4DRadSet_infos_train = get_TJ4DRadSet_image_info(data_path, image_ids=train_img_ids)
    _calculate_num_points_in_gt(TJ4DRadSet_infos_train)
    filename = save_path / 'TJ4DRadSet_infos_train.pkl'
    print(f'Kitti info train file is saved to {filename}')
    mmcv.dump(TJ4DRadSet_infos_train, filename)

    TJ4DRadSet_infos_test = get_TJ4DRadSet_image_info(data_path, image_ids=test_img_ids)
    _calculate_num_points_in_gt(TJ4DRadSet_infos_test)
    filename = save_path / 'TJ4DRadSet_infos_test.pkl'
    print(f'Kitti info test file is saved to {filename}')
    mmcv.dump(TJ4DRadSet_infos_test, filename)


def _create_reduced_point_cloud(info_path):
    """Create reduced point clouds for given info.

    Args:
        info_path (str): Path of data info.
    """
    TJ4DRadSet_infos = mmcv.load(info_path)

    for info in mmcv.track_iter_progress(TJ4DRadSet_infos):
        radar_info = info['radar_point_cloud']
        image_info = info['image']
        calib = info['calib']

        radar_path = Path(radar_info['velodyne_path'])
        radar_features = radar_info['num_features']
        points_radar = np.fromfile(str(radar_path), dtype=np.float32, count=-1).reshape([-1, radar_features])

        rect = calib['R0_rect']
        P0 = calib['P0']
        Trr2c = calib['Tr_radar_to_cam']

        points_radar = box_np_ops.remove_outside_points(points_radar, rect, Trr2c, P0, image_info['image_shape'])
        save_dir_radar = radar_path.parent.parent / (radar_path.parent.stem + '_reduced')
        if not save_dir_radar.exists():
            save_dir_radar.mkdir()
        save_filename_radar = save_dir_radar / radar_path.name

        with open(save_filename_radar, 'w') as f:
            points_radar.tofile(f)


def create_reduced_point_cloud(data_path):
    """Create reduced point clouds for training/testing.

    Args:
        data_path (str): Path of original data.
    """
    train_info_path = Path(data_path) / 'TJ4DRadSet_infos_train.pkl'
    test_info_path = Path(data_path) / 'TJ4DRadSet_infos_test.pkl'

    print('create reduced point cloud for training set')
    _create_reduced_point_cloud(train_info_path)
    print('create reduced point cloud for testing set')
    _create_reduced_point_cloud(test_info_path)
