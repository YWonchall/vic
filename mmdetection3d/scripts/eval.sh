CONFIG='/workspace/vic-competition/dair-v2x/configs/vic3d/filted-early-fusion/pointcloud/pointpillars/trainval_config_veh_1.py'
CHECKPOINT="/workspace/vic-competition/dair-v2x/configs/vic3d/filted-early-fusion/pointcloud/pointpillars/pointpillars_veh_1_vic_coop.pth"
python tools/test.py $CONFIG $CHECKPOINT --eval mAP