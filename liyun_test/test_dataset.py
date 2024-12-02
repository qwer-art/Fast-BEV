# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp
import sys
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(project_root)

mmdet3d_root = os.environ.get('MMDET3D')
if mmdet3d_root is not None and osp.exists(mmdet3d_root):
    import sys
    sys.path.insert(0, mmdet3d_root)
    print(f"using mmdet3d: {mmdet3d_root}")

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
from IPython import embed
import ipdb

def main():
    config_path = "configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py"
    cfg = Config.fromfile(config_path)
    data_train = cfg.data.train
    print(f"data_train_type: {data_train['type']}")
    datasets = [build_dataset(data_train)]

if __name__ == '__main__':
    main()