name2id = {
    "car": 2,
    "van": 2,
    "truck": 2,
    "bus": 2,
    "cyclist": 1,
    "tricyclist": 3,
    "motorcyclist": 3,
    "barrow": 3,
    "barrowlist": 3,
    "pedestrian": 0,
    "trafficcone": 3,
    "pedestrianignore": 3,
    "carignore": 3,
    "otherignore": 3,
    "unknowns_unmovable": 3,
    "unknowns_movable": 3,
    "unknown_unmovable": 3,
    "unknown_movable": 3,
}
# 3
superclass = {
    -1: "ignore",
    0: "pedestrian",
    1: "cyclist",
    2: "car",
    3: "Trafficcone",
    4: "ignore",
}


def add_arguments(parser): 
    parser.add_argument("--n", type=float, default=1) # 过滤范围
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--test", action="store_true") # 仅推理
    # 仅对early fusion有效
    parser.add_argument("--inf-filter", action="store_true") # 路端点云过滤
    parser.add_argument("--set-inf-label", action="store_true") # 设置预测类别为2(单类)
    parser.add_argument("--set-veh-label", action="store_true") # 设置预测类别为2(单类)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--split-data-path", type=str, default="../data/split_datas/example-cooperative-split-data.json"
    )
    parser.add_argument("--dataset", type=str, default="vic-sync")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--pred-classes", nargs="+", default=["car"])
    parser.add_argument("--model", type=str, default="veh_only")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-point-cloud", action="store_true")
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--save-multimodal", action="store_true")
    parser.add_argument("--extended-range", type=float, nargs="+", default=[-10, -49.68, -3, 79.12, 49.68, 1])
    parser.add_argument("--sensortype", type=str, default="lidar")
    parser.add_argument("--inf-sensortype", type=str, default="lidar")
    parser.add_argument("--veh-sensortype", type=str, default="lidar")
    parser.add_argument("--eval-single", action="store_true")
