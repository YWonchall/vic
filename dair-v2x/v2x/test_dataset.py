import sys
sys_path = [
    '/workspace/dair-v2x/v2x',
    '/workspace/dair-v2x/v2x/dataset',
    '/workspace/dair-v2x/v2x/dataset/dataset_utils'
]
sys.path += sys_path
import os.path as osp
from functools import cmp_to_key
import logging

logger = logging.getLogger(__name__)

from dataset.base_dataset import DAIRV2XDataset, get_annos, build_path_to_info
from dataset.dataset_utils import load_json, InfFrame, VehFrame, VICFrame, Label
from v2x_utils import Filter, RectFilter, id_cmp, id_to_str, get_trans, box_translation

from tqdm import tqdm
import numpy as np
from dataset import SUPPROTED_DATASETS
VICSyncDataset = SUPPROTED_DATASETS["vic-sync"]

input = "../data/DAIR-V2X/cooperative-vehicle-infrastructure-example/"
split = "val"
sensortype = "camera"
box_range = np.array([-10, -49.68, -3, 79.12, 49.68, 1])
indexs = [
    [0, 1, 2],
    [3, 1, 2],
    [3, 4, 2],
    [0, 4, 2],
    [0, 1, 5],
    [3, 1, 5],
    [3, 4, 5],
    [0, 4, 5],
]
extended_range = np.array([[box_range[index] for index in indexs]])


args = {
    'output': './',
    'model': 'late_fusion',
    'split_data_path': '/workspace/dair-v2x/data/split_datas/example-cooperative-split-data.json',
    'device': 0

}


dataset = VICSyncDataset(
        path=input, 
        args=args,
        split=split,
        sensortype=sensortype,
        extended_range=extended_range
    )

for VICFrame_data, label, filt in tqdm(dataset):
    veh_image_path = VICFrame_data.vehicle_frame()["image_path"][-10:-4]
    inf_image_path = VICFrame_data.infrastructure_frame()["image_path"][-10:-4]
    print(veh_image_path, inf_image_path)
    break
