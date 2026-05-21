import torch
from PIL import Image
from numpy import random
import numpy as np

from mmdet3d.datasets.pipelines import GlobalRotScaleTrans
from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class PointsRangeFilterV2(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range, sensor='lidar'):
        assert sensor in ['lidar', 'radar']
        self.sensor = sensor
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        sensor = self.sensor
        points = input_dict[f'{sensor}_points']

        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = input_dict[f'{sensor}_points'][points_mask]
        input_dict[f'{sensor}_points'] = clean_points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class PointShuffleV2(object):
    """Shuffle input points."""

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        if 'lidar_points' in input_dict:
            input_dict['lidar_points'].shuffle()
        if 'radar_points' in input_dict:
            input_dict['radar_points'].shuffle()

        return input_dict

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class RandomFlip3DV2:
    def __init__(self, flip_ratio_bev_horizontal=0.0, flip_ratio_bev_vertical=0.0):
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(flip_ratio_bev_horizontal, (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(flip_ratio_bev_vertical, (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str, optional): Flip direction.
                Default: 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        # for semantic segmentation task, only points will be flipped.
        if 'radar_points' in input_dict:
            input_dict['radar_points'].flip(direction)
        if 'lidar_points' in input_dict:
            input_dict['lidar_points'].flip(direction)

        if 'bbox3d_fields' in input_dict:
            if len(input_dict['bbox3d_fields']) == 0:  # test mode
                input_dict['bbox3d_fields'].append('empty_box3d')
                input_dict['empty_box3d'] = input_dict['box_type_3d'](np.array([], dtype=np.float32))

            assert len(input_dict['bbox3d_fields']) == 1
            for key in input_dict['bbox3d_fields']:
                input_dict[key].flip(direction)


    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        if 'pcd_horizontal_flip' not in input_dict:
            flip_horizontal = True if np.random.rand() < self.flip_ratio_bev_horizontal else False
            input_dict['pcd_horizontal_flip'] = flip_horizontal
        if 'pcd_vertical_flip' not in input_dict:
            flip_vertical = True if np.random.rand() < self.flip_ratio_bev_vertical else False
            input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(flip_ratio_bev_horizontal={self.flip_ratio_bev_horizontal},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str


@PIPELINES.register_module()
class GlobalRotScaleTransV2(GlobalRotScaleTrans):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float], optional): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float], optional): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float], optional): The standard deviation of
            translation noise applied to a scene, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool, optional): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """
    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        if 'radar_points' in input_dict:
            input_dict['radar_points'].translate(trans_factor)
        if 'lidar_points' in input_dict:
            input_dict['lidar_points'].translate(trans_factor)

        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)


    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        if 'radar_points' in input_dict:
            rot_mat_T = input_dict['radar_points'].rotate(noise_rotation)
        if 'lidar_points' in input_dict:
            rot_mat_T = input_dict['lidar_points'].rotate(noise_rotation)

        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                input_dict[key].rotate(noise_rotation)

        input_dict['pcd_rotation'] = rot_mat_T
        input_dict['pcd_rotation_angle'] = noise_rotation


    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']

        if 'radar_points' in input_dict:
            points = input_dict['radar_points']
            points.scale(scale)
            if self.shift_height:
                assert 'height' in points.attribute_dims.keys(), 'setting shift_height=True but points have no height attribute'
                points.tensor[:, points.attribute_dims['height']] *= scale
            input_dict['radar_points'] = points

        if 'lidar_points' in input_dict:
            points = input_dict['lidar_points']
            points.scale(scale)
            if self.shift_height:
                assert 'height' in points.attribute_dims.keys(), 'setting shift_height=True but points have no height attribute'
                points.tensor[:, points.attribute_dims['height']] *= scale
            input_dict['lidar_points'] = points

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)


@PIPELINES.register_module()
class RemoveDuplicatePoints(object):
    def __init__(self, sensor='radar', atol=1e-4):
        self.sensor = sensor
        self.atol = atol

    def __call__(self, results):
        if f'{self.sensor}_points' in results:
            points = results[f'{self.sensor}_points'].tensor
            points = self.remove_duplicate_points(points)
            results[f'{self.sensor}_points'].tensor = points
        return results

    def remove_duplicate_points(self, points):
        coords = points[:, :3]
        if self.atol is not None:
            coords = torch.round(coords / self.atol)

        unique_coords, inverse_indices, counts = torch.unique(
            coords,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )  # unique_coords: (M, 3), inverse_indices: (N,), counts: (M,)

        new_points = torch.zeros(unique_coords.shape[0], points.shape[1], dtype=points.dtype, device=points.device)
        new_points.index_reduce_(dim=0, index=inverse_indices, source=points, reduce='mean', include_self=False)
        return new_points

