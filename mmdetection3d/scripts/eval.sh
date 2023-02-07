CONFIG='/workspace/vic-competition/dair-v2x/configs/vic3d/filted-early-fusion/pointcloud/second/trainval_config_inf_1.py'
CHECKPOINT="/workspace/vic-competition/dair-v2x/configs/vic3d/filted-early-fusion/pointcloud/second/second_inf_1_vic_inf.pth"
python tools/test.py $CONFIG $CHECKPOINT --eval mAP