import sys
import os
import os.path as osp

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import argparse
import logging

logger = logging.getLogger(__name__)

from tqdm import tqdm
import numpy as np
import pandas as pd

from v2x_utils import range2box, id_to_str, Evaluator, assign_gt_boxes
from config import add_arguments
from dataset import SUPPROTED_DATASETS
from dataset.dataset_utils import save_pkl
from models import SUPPROTED_MODELS
from models.model_utils import Channel


def eval_vic(args, dataset, model, evaluator):
    idx = -1
    data_arr = []
    for VICFrame, label, filt in tqdm(dataset):
        idx += 1
        # if idx % 10 != 0:
        #     continue
        try:
            veh_id = dataset.data[idx][0]["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")
        except Exception:
            veh_id = VICFrame["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")
        # 2.1
        pred, pred_dict = model(
            VICFrame,
            filt,
            None if not hasattr(dataset, "prev_inf_frame") else dataset.prev_inf_frame,
        )
        # prev_inf_frame用于async的路端
        #print(pred_dict)
        assigned_boxes = assign_gt_boxes(label,pred_dict,'car', 0.5, "bev")
        for box in assigned_boxes:
            veh_boxes = box['box'].reshape(-1)
            inf_boxes = box['inf_pred'].reshape(-1)
            gt_boxes = box['gt_boxes'].reshape(-1)
            veh_scores = box['score']
            inf_socres = box['inf_scores']
            data = np.concatenate((veh_boxes,inf_boxes,np.array([veh_scores]),np.array([inf_socres]),gt_boxes))
            data_arr.append(data)
        # evaluator.add_frame(pred, label)
        #data_arr = np.array(data_arr)
        # print(data_arr.shape)
        pipe.flush()
        
        #inf_pred = pred_dict.pop('inf_pred')
        # if idx == 5:
        #     pass
        # pred["label"] = label["boxes_3d"]
        # pred["veh_id"] = veh_id
        # save_pkl(pred, osp.join(args.output, "result", pred["veh_id"] + ".pkl"))
    pd.DataFrame(data_arr).to_csv(f'/workspace/late-fusion-data-train.csv',index=False)
    # evaluator.print_ap("3d")
    # evaluator.print_ap("bev")
    # print("Average Communication Cost = %.2lf Bytes" % (pipe.average_bytes()))


def eval_single(args, dataset, model, evaluator):
    for frame, label, filt in tqdm(dataset):
        pred = model(frame, filt)
        if args.sensortype == "camera":
            evaluator.add_frame(pred, label["camera"])
        elif args.sensortype == "lidar":
            evaluator.add_frame(pred, label["lidar"])
        save_pkl({"boxes_3d": label["lidar"]["boxes_3d"]}, osp.join(args.output, "result", frame.id["camera"] + ".pkl"))

    evaluator.print_ap("3d")
    evaluator.print_ap("bev")


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
    # 1.1
    # 1-1
    dataset = SUPPROTED_DATASETS[args.dataset](
        args.input,
        args,
        split=args.split,
        sensortype=args.sensortype,
        inf_sensortype=args.inf_sensortype,
        veh_sensortype=args.veh_sensortype,
        extended_range=extended_range,
    )
    

    logger.info("loading evaluator")
    evaluator = Evaluator(args.pred_classes)

    logger.info("loading model")
    if args.eval_single:
        model = SUPPROTED_MODELS[args.model](args)
        eval_single(args, dataset, model, evaluator)
    else:
        pipe = Channel()
        # 融合相关代码
        # 2-1
        model = SUPPROTED_MODELS[args.model](args, pipe)
        eval_vic(args, dataset, model, evaluator)
