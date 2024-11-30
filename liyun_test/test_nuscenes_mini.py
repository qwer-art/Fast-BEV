from os import path as osp
import sys
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
print(f"project_root: {project_root}")
sys.path.append(project_root)
from tools.data_converter import nuscenes_converter as nuscenes_converter

dataset = 'nuscenes'
version = 'v1.0-mini'
root_path = './data/nuscenes'
out_dir = './data/nuscenes'
info_prefix = 'nuscenes'
max_sweeps=10

if __name__ == '__main__':
    ### [1].data_infos
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)