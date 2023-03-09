import sys
import os
import os.path as osp

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import argparse
import logging

logger = logging.getLogger(__name__)

from tqdm import tqdm
import numpy as np
import json

from v2x_utils import range2box, id_to_str, Evaluator
from config import add_arguments
from dataset import SUPPROTED_DATASETS
from dataset.dataset_utils import save_pkl
from models import SUPPROTED_MODELS
from models.model_utils import Channel

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def eval_vic(args, dataset, model):
    idx = -1
    for VICFrame, _, filt in tqdm(dataset):
        idx += 1
        try:
            veh_id = dataset.data[idx][0]["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")
        except Exception:
            veh_id = VICFrame["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")
        pred = model(
            VICFrame,
            filt,
            None if not hasattr(dataset, "prev_inf_frame") else dataset.prev_inf_frame,
        )
        # 单类推理开启
        for ii in range(len(pred["labels_3d"])):
            pred["labels_3d"][ii] = 2
        
        outfile = osp.join(args.output, "result", veh_id + '.json')
        if(args.model=='late_fusion'):
            pred.pop('inf_id')
            pred.pop('veh_id')
            pred.pop('inf_boxes')
        
        pred['ab_cost'] = pipe.cur_bytes
        with open(outfile,'w') as f:
            json.dump(pred,f,cls=NumpyEncoder)
        pipe.flush()
    print("Average Communication Cost = %.2lf Bytes" % (pipe.average_bytes()))


def eval_single(args, dataset, model):
    for frame, label, filt in tqdm(dataset):
        pred = model(frame, filt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    add_arguments(parser)
    args, _ = parser.parse_known_args()
    # add model-specific arguments
    SUPPROTED_MODELS[args.model].add_arguments(parser)
    args = parser.parse_args()

    if args.quiet:
        level = logging.ERROR
    elif args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

    extended_range = range2box(np.array(args.extended_range))
    logger.info("loading dataset")

    dataset = SUPPROTED_DATASETS[args.dataset](
        args.input,
        args,
        split=args.split,
        sensortype=args.sensortype,
        inf_sensortype=args.inf_sensortype,
        veh_sensortype=args.veh_sensortype,
        extended_range=extended_range,
    )

    logger.info("loading model")
    if args.eval_single:
        model = SUPPROTED_MODELS[args.model](args)
        eval_single(args, dataset, model)
    else:
        pipe = Channel()
        # 融合相关代码
        model = SUPPROTED_MODELS[args.model](args, pipe)
        eval_vic(args, dataset, model)
