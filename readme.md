# RadarGaussianDet3D: Gaussian Representation-based Real-time 3D Object Detection with 4D Automotive Radars

Paper: [IEEE](https://ieeexplore.ieee.org/document/11434902), [arXiv](https://arxiv.org/abs/2509.16119)

## Usage

### Prerequisites

We list our environment setup below:

* Python 3.8
* PyTorch 1.12.0+cu113
* MMCV 1.6.0
* MMDetection 2.25.0
* MMSegmentation 0.26.0
* MMDetection3D 1.0.0rc3
* vod-tudelft 1.0.3  (This is the toolkit of vod dataset, and can be installed by `pip install vod-tudelft==1.0.3` command.)

After setting up the environment, please move the files in this repo to your mmdetection3d folder.

After that, run the following command to build the CUDA extensions for Gaussian Splatting:
```
cd plugin/RadarGaussianDet3D/ops/diff-gaussian-rasterization-bev
python setup.py install
cd ../../../..
```

### Data Preparation

Please use the file provided in `tools/create_data_for_RadarGaussianDet3D.py` to generate the corresponding data.

```
python tools/create_data_for_RadarGaussianDet3D.py --dataset vod --root-path ${YOUR_DATA_PATH}$
```

Please also make sure you edit the `data_root` in `plugin/RadarGaussianDet3D/configs/_base_/datasets/vod_radar.py` and 
`plugin/RadarGaussianDet3D/configs/_base_/datasets/tj4d_radar.py` to point to the correct data directory.

### Train

To train RadarGaussianDet3D with a single GPU, you can use the following command:

```
python tools/train_v2.py plugin/RadarGaussianDet3D/configs/gaussian_sec_vod_radar_centerhead.py
```

### Evaluation

To evaluate the trained model, you can use the following command:

```
python tools/test_v2.py plugin/RadarGaussianDet3D/configs/gaussian_sec_vod_radar_centerhead.py trained_models/RadarGaussianDet3D_vod.pth --eval bbox
```

## Citation
```
@article{RadarGaussianDet3D,
    author={Xiong, Weiyi and Zhu, Bing and Zheng, Zewei},
    journal={IEEE Robotics and Automation Letters}, 
    title={RadarGaussianDet3D: Gaussian Representation-Based Real-Time 3D Object Detection With 4D Automotive Radars}, 
    year={2026},
    volume={11},
    number={5},
    pages={5709-5716}
}
```
