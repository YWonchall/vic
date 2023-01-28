CONFIG='/workspace/vic-competition/dair-v2x/configs/vic3d/late-fusion-pointcloud/pointpillars/trainval_config_v_car.py'
CHECKPOINT="/workspace/vic-competition/dair-v2x/configs/vic3d/late-fusion-pointcloud/pointpillars/pointpillars_veh_my_car.pth"
python tools/test.py $CONFIG $CHECKPOINT --eval mAP