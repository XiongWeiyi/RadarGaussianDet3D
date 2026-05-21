from mmcv.utils import print_log
from mmdet3d.datasets.builder import DATASETS
from .vod_dataset import VoDDataset
from ..core.evaluation import TJ4DRadSet_eval


@DATASETS.register_module()
class TJ4DRadDataset(VoDDataset):
    CLASSES = ('Car', 'Pedestrian', 'Cyclist', 'Truck', 'Other')

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
                Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)

        gt_annos = [info['annos'] for info in self.data_infos]

        ap_dict = dict()

        ap_result_str, ap_dict_ = TJ4DRadSet_eval(gt_annos, result_files['pts_bbox'], self.CLASSES, custom_method=0)  # 考虑所有边界框
        # ap_result_str_valid, ap_dict_valid_ = TJ4DRadSet_eval(gt_annos, result_files['pts_bbox'], self.CLASSES, custom_method=1)  # 仅考虑有点的边界框

        for ap_type, ap in ap_dict_.items():
            ap_dict[f'pts_bbox/all_bbox/{ap_type}'] = float('{:.4f}'.format(ap))
        # for ap_type, ap in ap_dict_valid_.items():
        #     ap_dict[f'pts_bbox/valid_bbox/{ap_type}'] = float('{:.4f}'.format(ap))
        print_log('Results of all bboxes:\n' + ap_result_str, logger=logger)
        # print_log('Results of valid bboxes:\n' + ap_result_str_valid, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict
