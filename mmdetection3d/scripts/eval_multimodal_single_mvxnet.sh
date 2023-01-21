CONFIG="../dair-v2x/configs/vic3d/late-fusion-multimodal/mvxnet/trainval_config_v.py"
CHECKPOINT="../dair-v2x/configs/vic3d/late-fusion-multimodal/mvxnet/sv3d_veh_mvxnet_bf0e32c42649ee90e03f937214356dbf.pth"
python tools/test.py $CONFIG $CHECKPOINT --eval mAP