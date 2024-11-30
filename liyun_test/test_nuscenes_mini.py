from os import path as osp
import sys
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(project_root)
from tools.data_converter import nuscenes_converter as nuscenes_converter
import mmcv

dataset = 'nuscenes'
version = 'v1.0-mini'
root_path = './data/nuscenes'
out_dir = './data/nuscenes'
info_prefix = 'nuscenes'
max_sweeps=10

if __name__ == '__main__':
    ### [1].data_infos
    # nuscenes_converter.create_nuscenes_infos(
    #     root_path, info_prefix, version=version, max_sweeps=max_sweeps)
    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    nuscenes_converter.export_2d_annotation(
        root_path, info_train_path, version=version)
    nuscenes_converter.export_2d_annotation(
        root_path, info_val_path, version=version)
