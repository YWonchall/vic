CONFIG="/workspace/vic-competition/dair-v2x/configs/vic3d/early-fusion-multimodal/mvxnet/trainval_config.py"
WORKDIR="/workspace/vic-competition/mmdetection3d/work-dirs/vic/veh/early-fusion/mvxnet/train"
python tools/train.py $CONFIG  --work-dir $WORKDIR