import argparse
from tools.data_converter import vod_converter as vod
from tools.data_converter import TJ4DRadSet_converter as TJ4DRadSet

def data_prep(dataset, root_path):            # .../view_of_delft_PUBLIC  or  .../TJ4DRadSet
    """Prepare data related to View-of-Delft or TJ4DRadSet dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
    """
    if dataset == 'vod':
        vod.create_vod_info_file(root_path)
        vod.create_reduced_point_cloud(root_path)
    elif dataset == 'TJ4DRadSet':
        TJ4DRadSet.create_TJ4DRadSet_info_file(root_path)
        TJ4DRadSet.create_reduced_point_cloud(root_path)
    else:
        raise NotImplementedError


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--dataset', type=str, default='vod', help='dataset type (vod or TJ4DRadSet)')
parser.add_argument('--root-path', type=str, help='specify the root path of dataset')
args = parser.parse_args()

if __name__ == '__main__':
    data_prep(dataset=args.dataset, root_path=args.root_path)
