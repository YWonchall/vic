CONFIG="/workspace/vic-competition/dair-v2x/configs/vic3d-report/late-fusion/pointpillars/trainval_config_veh_1.py"
WORKDIR="/workspace/vic-competition/mmdetection3d/work-dirs/vic-report/pointpillars/pointpillars-veh"

python tools/train.py $CONFIG  --work-dir $WORKDIR