CONFIG="/workspace/vic-competition/dair-v2x/configs/vic3d/late-fusion-pointcloud/pointpillars/trainval_config_v_car.py"
WORKDIR="./work-dirs/vic/pointpillars/veh-car/train"
python tools/train.py $CONFIG  --work-dir $WORKDIR