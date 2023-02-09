CONFIG="/workspace/vic-competition/dair-v2x/configs/vic3d/filted-early-fusion/pointcloud/second/trainval_config_veh_1.py"
WORKDIR="/workspace/vic-competition/mmdetection3d/work-dirs/vic-coop-all/train/second-car"

python tools/train.py $CONFIG  --work-dir $WORKDIR