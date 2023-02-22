import os.path as osp
import sys
import os
import numpy as np
import torch.nn as nn
import logging
import json
import open3d as o3d
logger = logging.getLogger(__name__)

from base_model import BaseModel
from model_utils import (
    init_model,
    inference_detector,
    inference_mono_3d_detector,
    inference_multi_modality_detector,
    BBoxList,
    EuclidianMatcher,
    SpaceCompensator,
    TimeCompensator,
    BasicFuser,
    read_pcd,
    concatenate_pcd2bin,
    filt_point_by_boxes,
)
from dataset.dataset_utils import (
    load_json,
    save_pkl,
    load_pkl,
    read_jpg,
)
from v2x_utils import (
    mkdir,
    get_arrow_end,
    box_translation,
    points_translation,
    get_trans,
    diff_label_filt,
)
# class NumpyEncoder(json.JSONEncoder):
#     """ Special json encoder for numpy types """
#     def default(self, obj):
#         if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
#                             np.int16, np.int32, np.int64, np.uint8,
#                             np.uint16, np.uint32, np.uint64)):
#             return int(obj)
#         elif isinstance(obj, (np.float_, np.float16, np.float32,
#                               np.float64)):
#             return float(obj)
#         elif isinstance(obj, (np.ndarray,)):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)

def get_box_info(result):
    if len(result[0]["boxes_3d"].tensor) == 0:
        box_lidar = np.zeros((1, 8, 3))
        box_ry = np.zeros(1)
    else:
        box_lidar = result[0]["boxes_3d"].corners.numpy()
        box_ry = result[0]["boxes_3d"].tensor[:, -1].numpy()
    box_centers_lidar = box_lidar.mean(axis=1)
    arrow_ends_lidar = get_arrow_end(box_centers_lidar, box_ry)
    return box_lidar, box_ry, box_centers_lidar, arrow_ends_lidar


def gen_pred_dict(id, timestamp, box, arrow, points, score, label):
    if len(label) == 0:
        score = [-2333]
        label = [-1]
    save_dict = {
        "info": id,
        "timestamp": timestamp,
        "boxes_3d": box.tolist(),
        "arrows": arrow.tolist(),
        "scores_3d": score,
        "labels_3d": label,
        "points": points.tolist(),
    }
    return save_dict


class EarlyFusion(BaseModel):
    def add_arguments(parser):
        parser.add_argument("--inf-config-path", type=str, default="")
        parser.add_argument("--inf-model-path", type=str, default="")
        parser.add_argument("--veh-config-path", type=str, default="")
        parser.add_argument("--veh-model-path", type=str, default="")
        parser.add_argument("--no-comp", action="store_true")
        parser.add_argument("--overwrite-cache", action="store_true")

    def __init__(self, args, pipe):
        super().__init__()
        self.inf_model = LateFusionInf(args, pipe)
        self.model = LateFusionVeh(args)
        self.args = args
        self.pipe = pipe
        mkdir(args.output)
        mkdir(osp.join(args.output, "inf"))
        mkdir(osp.join(args.output, "veh"))
        mkdir(osp.join(args.output, "inf", "lidar"))
        mkdir(osp.join(args.output, "veh", "lidar"))
        mkdir(osp.join(args.output, "inf", "camera"))
        mkdir(osp.join(args.output, "veh", "camera"))
        mkdir(osp.join(args.output, "result"))

    def forward(self, vic_frame, filt, prev_inf_frame_func=None, *args):
        save_path = osp.join(vic_frame.path, "vehicle-side", "cache")
        if not osp.exists(save_path):
            mkdir(save_path)
        name = vic_frame.veh_frame["image_path"][-10:-4]
        Inf_points = read_pcd(osp.join(vic_frame.path, "infrastructure-side", vic_frame.inf_frame["pointcloud_path"]))
        Veh_points = read_pcd(osp.join(vic_frame.path, "vehicle-side", vic_frame.veh_frame["pointcloud_path"]))
        
        
        # TODO
        # 此处做过滤获得新的Inf_points
        if self.args.inf_filter:
            # 路端lidar坐标下的boxes 8点
            # print("— using inf filter —")
            pred_inf,id = self.inf_model(
                vic_frame.infrastructure_frame(),
                None, # vic_frame.transform(from_coord="Infrastructure_lidar", to_coord="Vehicle_lidar"),
                filt,
                prev_inf_frame_func if not self.args.no_comp else None,
            )
            if self.args.set_inf_label:
                for ii in range(len(pred_inf["labels_3d"])):
                    pred_inf["labels_3d"][ii] = 2
            Inf_points = filt_point_by_boxes(Inf_points, pred_inf, 2,self.args.n)
            # outfile = '/workspace/demo.json'
            # print(id)
            # with open(outfile,'w') as f:
            #     json.dump(pred_inf,f,cls=NumpyEncoder)


        # 路端坐标转车端坐标
        vic_frame_trans = vic_frame.transform(from_coord="Infrastructure_lidar", to_coord="Vehicle_lidar")
        for i in range(len(Inf_points.pc_data)):
            temp = vic_frame_trans.single_point_transformation(
                [Inf_points.pc_data[i][0], Inf_points.pc_data[i][1], Inf_points.pc_data[i][2]]
            )
            for j in range(3):
                Inf_points.pc_data[i][j] = temp[j]
            Inf_points.pc_data[i][3] = Inf_points.pc_data[i][3] * 255
        
        # 传送至车端与车端融合
        self.pipe.send("inf_points", Inf_points.pc_data)
        concatenate_pcd2bin(Inf_points, Veh_points, osp.join(save_path, name + ".pcd"))
        vic_frame.veh_frame["pointcloud_path"] = osp.join("cache", name + ".pcd")
        pred, id_veh = self.model(vic_frame.vehicle_frame(), None, filt)

        if self.args.set_veh_label:
            for ii in range(len(pred["labels_3d"])):
                pred["labels_3d"][ii] = 2
        # 使用单类模型融合推理时需要开启
        # Hard Code to change the prediction label
        # for ii in range(len(pred["labels_3d"])):
        #     pred["labels_3d"][ii] = 2

        # self.pipe.send("boxes_3d", pred["boxes_3d"])
        # self.pipe.send("labels_3d", pred["labels_3d"])
        # self.pipe.send("scores_3d", pred["scores_3d"])

        return {
            "boxes_3d": np.array(pred["boxes_3d"]),
            "labels_3d": np.array(pred["labels_3d"]),
            "scores_3d": np.array(pred["scores_3d"]),
        }

class LateFusionVeh(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = None
        self.args = args

    def pred(self, frame, trans, pred_filter):
        if self.args.veh_sensortype == "lidar":
            id = frame.id["lidar"]
            logger.debug("vehicle pointcloud_id: {}".format(id))
            path = osp.join(self.args.output, "veh", "lidar", id + ".pkl")
            frame_timestamp = frame["pointcloud_timestamp"]
        elif self.args.veh_sensortype == "camera":
            id = frame.id["camera"]
            logger.debug("vehicle image_id: {}".format(id))
            path = osp.join(self.args.output, "veh", "camera", id + ".pkl")
            frame_timestamp = frame["image_timestamp"]
        elif self.args.veh_sensortype == "multimodal":
            id = frame.id["lidar"]
            logger.debug("vehicle pointcloud_id: {}".format(id))
            path = osp.join(self.args.output, "veh", "multimodal", id + ".pkl")
            frame_timestamp = frame["pointcloud_timestamp"]

        if osp.exists(path) and not self.args.overwrite_cache:
            pred_dict = load_pkl(path)
            return pred_dict, id

        logger.debug("prediction not found, predicting...")
        if self.model is None:
            raise Exception

        if self.args.veh_sensortype == "lidar":
            tmp = frame.point_cloud(data_format="file")
            result, _ = inference_detector(self.model, tmp)
        elif self.args.veh_sensortype == "camera":
            tmp = osp.join(self.args.input, "vehicle-side", frame["image_path"])
            annos = osp.join(self.args.input, "vehicle-side", "annos", id + ".json")
            result, _ = inference_mono_3d_detector(self.model, tmp, annos)
        elif self.args.veh_sensortype == "multimodal":
            pcd_tmp = frame.point_cloud(data_format="file")
            img_tmp = osp.join(self.args.input, "vehicle-side", frame["image_path"])
            annos = osp.join(self.args.input, "vehicle-side", "annos", id + ".json")
            result, _ = inference_multi_modality_detector(self.model, pcd_tmp, img_tmp, annos)
            result = [result[0]['pts_bbox']]
            # result, _ = inference_mono_3d_detector(self.model, tmp, annos)
        
        box, box_ry, box_center, arrow_ends = get_box_info(result)

        # Convert to other coordinate
        if trans is not None:
            box = trans(box)
            box_center = trans(box_center)[:, np.newaxis, :]
            arrow_ends = trans(arrow_ends)[:, np.newaxis, :]

        # Filter out labels
        remain = []
        if len(result[0]["boxes_3d"].tensor) != 0:
            for i in range(box.shape[0]):
                if pred_filter(box[i]):
                    remain.append(i)

        # hard code by yuhb
        # TODO: new camera model
        if self.args.veh_sensortype == "camera":
            for ii in range(len(result[0]["labels_3d"])):
                result[0]["labels_3d"][ii] = 2

        if len(remain) >= 1:
            box = box[remain]
            box_center = box_center[remain]
            arrow_ends = arrow_ends[remain]
            result[0]["scores_3d"] = result[0]["scores_3d"].numpy()[remain]
            result[0]["labels_3d"] = result[0]["labels_3d"].numpy()[remain]
        else:
            box = np.zeros((1, 8, 3))
            box_center = np.zeros((1, 1, 3))
            arrow_ends = np.zeros((1, 1, 3))
            result[0]["labels_3d"] = np.zeros((1))
            result[0]["scores_3d"] = np.zeros((1))

        if self.args.veh_sensortype == "lidar" and self.args.save_point_cloud:
            save_data = trans(frame.point_cloud(format="array"))
        elif self.args.veh_sensortype == "camera" and self.args.save_image:
            save_data = frame.image(data_format="array")
        elif self.args.veh_sensortype == "multimodal" and self.args.save_multimodal:
            save_data = trans(frame.point_cloud(format="array"))
            # save_image = frame.image(data_format="array")
            # save_data = [save_point_cloud, save_image]
        else:
            save_data = np.array([])

        pred_dict = gen_pred_dict(
            id,
            frame_timestamp,
            box,
            np.concatenate([box_center, arrow_ends], axis=1),
            save_data,
            result[0]["scores_3d"].tolist(),
            result[0]["labels_3d"].tolist(),
        )
        # save_pkl(pred_dict, path)

        return pred_dict, id

    def forward(self, data, trans, pred_filter):
        try:
            pred_dict, id = self.pred(data, trans, pred_filter)
        except Exception:
            logger.info("building model")
            self.model = init_model(
                self.args.veh_config_path,
                self.args.veh_model_path,
                device=self.args.device,
            )
            pred_dict, id = self.pred(data, trans, pred_filter)
        return pred_dict, id
'''
class LateFusionVeh(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = None
        self.args = args
        self.args.overwrite_cache = True

    def pred(self, frame, trans, pred_filter):
        if self.args.sensortype == "lidar":
            id = frame.id["lidar"]
            logger.debug("vehicle pointcloud_id: {}".format(id))
            path = osp.join(self.args.output, "veh", "lidar", id + ".pkl")
            frame_timestamp = frame["pointcloud_timestamp"]

            if osp.exists(path) and self.args.overwrite_cache:
                pred_dict = load_pkl(path)
                return pred_dict, id

            logger.debug("predicting...")
            if self.model is None:
                raise Exception

            tmp = frame.point_cloud(data_format="file")
            result, _ = inference_detector(self.model, tmp)
            box, box_ry, box_center, arrow_ends = get_box_info(result)
            if trans is not None:
                box = trans(box)  #
                box_center = trans(box_center)[:, np.newaxis, :]
                arrow_ends = trans(arrow_ends)[:, np.newaxis, :]

            remain = []
            if len(result[0]["boxes_3d"].tensor) != 0:
                for i in range(box.shape[0]):
                    if pred_filter(box[i]):
                        remain.append(i)

            if len(remain) >= 1:
                box = box[remain]
                box_center = box_center[remain]
                arrow_ends = arrow_ends[remain]
                result[0]["scores_3d"] = result[0]["scores_3d"].numpy()[remain]
                result[0]["labels_3d"] = result[0]["labels_3d"].numpy()[remain]
            else:
                box = np.zeros((1, 8, 3))
                box_center = np.zeros((1, 1, 3))
                arrow_ends = np.zeros((1, 1, 3))
                result[0]["labels_3d"] = np.zeros((1))
                result[0]["scores_3d"] = np.zeros((1))

            if self.args.save_point_cloud:
                save_data = frame.point_cloud(format="array")
            else:
                save_data = np.array([])

            pred_dict = gen_pred_dict(
                id,
                frame_timestamp,
                box,
                np.concatenate([box_center, arrow_ends], axis=1),
                save_data,
                result[0]["scores_3d"].tolist(),
                result[0]["labels_3d"].tolist(),
            )
            save_pkl(pred_dict, path)

            return pred_dict, id
        else:
            print("Now early fusion only supports LiDAR sensor!")
            raise Exception

    def forward(self, data, trans, pred_filter):
        try:
            pred_dict, id = self.pred(data, trans, pred_filter)
        except Exception:
            logger.info("building model")
            self.model = init_model(
                self.args.veh_config_path,
                self.args.veh_model_path,
                device=self.args.device,
            )
            pred_dict, id = self.pred(data, trans, pred_filter)

        return pred_dict, id
'''
class LateFusionInf(nn.Module):
    def __init__(self, args, pipe):
        super().__init__()
        self.model = None
        self.args = args
        self.pipe = pipe

    def pred(self, frame, trans, pred_filter):
        # 2.4
        # 2-4
        if self.args.inf_sensortype == "lidar":
            id = frame.id["lidar"]
            logger.debug("infrastructure pointcloud_id: {}".format(id))
            path = osp.join(self.args.output, "inf", "lidar", id + ".pkl")
            frame_timestamp = frame["pointcloud_timestamp"]
        elif self.args.inf_sensortype == "camera":
            id = frame.id["camera"]
            logger.debug("infrastructure image_id: {}".format(id))
            path = osp.join(self.args.output, "inf", "camera", id + ".pkl")
            frame_timestamp = frame["image_timestamp"]
        elif self.args.inf_sensortype == "multimodal":
            id = frame.id["lidar"]
            logger.debug("infrastructure pointcloud_id: {}".format(id))
            path = osp.join(self.args.output, "inf", "multimodal", id + ".pkl")
            frame_timestamp = frame["pointcloud_timestamp"]

        if osp.exists(path) and not self.args.overwrite_cache:
            pred_dict = load_pkl(path)
            return pred_dict, id

        logger.debug("prediction not found, predicting...")
        if self.model is None:
            raise Exception
        # 2.5
        # 2-5
        if self.args.inf_sensortype == "lidar":
            tmp = frame.point_cloud(data_format="file")
            result, _ = inference_detector(self.model, tmp)
        elif self.args.inf_sensortype == "camera":
            tmp = osp.join(self.args.input, "infrastructure-side", frame["image_path"])
            annos = osp.join(self.args.input, "infrastructure-side", "annos", id + ".json")
            result, _ = inference_mono_3d_detector(self.model, tmp, annos)
        elif self.args.inf_sensortype == "multimodal":
            pcd_tmp = frame.point_cloud(data_format="file")
            img_tmp = osp.join(self.args.input, "infrastructure-side", frame["image_path"])
            annos = osp.join(self.args.input, "infrastructure-side", "annos", id + ".json")
            # 2.6
            result, _ = inference_multi_modality_detector(self.model, pcd_tmp, img_tmp, annos)
            result = [result[0]['pts_bbox']]
            # result, _ = inference_mono_3d_detector(self.model, tmp, annos)
        box, box_ry, box_center, arrow_ends = get_box_info(result)

        # Convert to other coordinate
        if trans is not None:
            box = trans(box)
            box_center = trans(box_center)[:, np.newaxis, :]
            arrow_ends = trans(arrow_ends)[:, np.newaxis, :]

        # Filter out labels
        remain = []
        if len(result[0]["boxes_3d"].tensor) != 0:
            for i in range(box.shape[0]):
                if pred_filter(box[i]):
                    remain.append(i)

        # hard code by yuhb
        # TODO: new camera model
        if self.args.inf_sensortype == "camera":
            for ii in range(len(result[0]["labels_3d"])):
                result[0]["labels_3d"][ii] = 2

        if len(remain) >= 1:
            box = box[remain]
            box_center = box_center[remain]
            arrow_ends = arrow_ends[remain]
            result[0]["scores_3d"] = result[0]["scores_3d"].numpy()[remain]
            result[0]["labels_3d"] = result[0]["labels_3d"].numpy()[remain]
        else:
            box = np.zeros((1, 8, 3))
            box_center = np.zeros((1, 1, 3))
            arrow_ends = np.zeros((1, 1, 3))
            result[0]["labels_3d"] = np.zeros((1))
            result[0]["scores_3d"] = np.zeros((1))

        if self.args.inf_sensortype == "lidar" and self.args.save_point_cloud:
            save_data = trans(frame.point_cloud(format="array"))
        elif self.args.inf_sensortype == "camera" and self.args.save_image:
            save_data = frame.image(data_format="array")
        elif self.args.inf_sensortype == "multimodal" and self.args.save_multimodal:
            save_data = trans(frame.point_cloud(format="array"))
            # save_image = frame.image(data_format="array")
            # save_data = [save_point_cloud, save_image]
        else:
            save_data = np.array([])

        pred_dict = gen_pred_dict(
            id,
            frame_timestamp,
            box,
            np.concatenate([box_center, arrow_ends], axis=1),
            save_data,
            result[0]["scores_3d"].tolist(),
            result[0]["labels_3d"].tolist(),
        )
        #save_pkl(pred_dict, path)

        return pred_dict, id

    def forward(self, data, trans, pred_filter, prev_inf_frame_func=None):
        # 2.3
        # data: inf_fram, trans: inf_lidar -> veh_lidar
        try:
            pred_dict, id = self.pred(data, trans, pred_filter)
        except Exception:
            logger.info("building model")
            self.model = init_model(
                self.args.inf_config_path,
                self.args.inf_model_path,
                device=self.args.device,
            )
            # 2.4
            # 2-3
            pred_dict, id = self.pred(data, trans, pred_filter)
        # self.pipe.send("boxes", pred_dict["boxes_3d"])
        # self.pipe.send("score", pred_dict["scores_3d"])
        # self.pipe.send("label", pred_dict["labels_3d"])

        if prev_inf_frame_func is not None:
            prev_frame, delta_t = prev_inf_frame_func(id, sensortype=self.args.inf_sensortype)
            if prev_frame is not None:
                prev_frame_trans = prev_frame.transform(from_coord="Infrastructure_lidar", to_coord="Vehicle_lidar")
                prev_frame_trans.veh_name = trans.veh_name
                prev_frame_trans.delta_x = trans.delta_x
                prev_frame_trans.delta_y = trans.delta_y
                try:
                    pred_dict, _ = self.pred(
                        prev_frame,
                        prev_frame_trans,
                        pred_filter,
                    )
                except Exception:
                    logger.info("building model")
                    self.model = init_model(
                        self.args.inf_config_path,
                        self.args.inf_model_path,
                        device=self.args.device,
                    )
                    pred_dict, _ = self.pred(
                        prev_frame,
                        prev_frame_trans,
                        pred_filter,
                    )
                # self.pipe.send("prev_boxes", pred_dict["boxes_3d"])
                # self.pipe.send("prev_time_diff", delta_t)
                # self.pipe.send("prev_label", pred_dict["labels_3d"])

        return pred_dict, id


if __name__ == "__main__":
    sys.path.append("..")
    sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
