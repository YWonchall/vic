CONFIG="../dair-v2x/configs/vic3d/late-fusion-multimodal/mvxnet/trainval_config_i.py"
CHECKPOINT="/workspace/vic-competition/mmdetection3d/checkpoints/mvxnet.pth"
python tools/test.py $CONFIG $CHECKPOINT --eval mAP