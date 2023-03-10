CONFIG="/home/cao/code/ywc/vic/dair-v2x/configs/vic3d/EFWF/pointpillars/trainval_config_veh_1.py"
WORKDIR="./work-dirs/test/"

python tools/train.py $CONFIG  --work-dir $WORKDIR