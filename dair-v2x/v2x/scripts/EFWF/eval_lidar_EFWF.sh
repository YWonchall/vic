# 1
DATA="../data/DAIR-V2X/cooperative-vehicle-infrastructure-example"
OUTPUT="../cache/vic-early-lidar"
rm -r $OUTPUT
# 2 使用单类需要注意类别处理
INFRA_MODEL_PATH="../configs/vic3d/EFWF/pointpillars"
INFRA_CONFIG_NAME="trainval_config_inf_1.py"
INFRA_MODEL_NAME="pointpillars_inf.pth"
# 3
VEHICLE_MODEL_PATH="../configs/vic3d/EFWF/pointpillars"
VEHICLE_CONFIG_NAME="trainval_config_veh_1.py"
VEHICLE_MODEL_NAME="pointpillars_veh.pth"
# 4
SPLIT_DATA_PATH="../data/split_datas/example-cooperative-split-data.json"

# srun --gres=gpu:a100:1 --time=1-0:0:0 --job-name "dair-v2x" \
CUDA_VISIBLE_DEVICES=0 #$1
FUSION_METHOD='EFWF' #$2
EXTEND_RANGE_START=0 #$4
EXTEND_RANGE_END=100 #$5
# sensortype用于读取数据集
# 5 sensortype
python eval.py \
  --n 3 \
  --input $DATA \
  --output $OUTPUT \
  --model $FUSION_METHOD \
  --dataset vic-sync \
  --inf-filter \
  --split val \
  --split-data-path $SPLIT_DATA_PATH \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/$INFRA_MODEL_NAME \
  --veh-config-path $VEHICLE_MODEL_PATH/$VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH/${VEHICLE_MODEL_NAME} \
  --device ${CUDA_VISIBLE_DEVICES} \
  --pred-class car \
  --inf-sensortype lidar \
  --veh-sensortype lidar \
  --extended-range $EXTEND_RANGE_START -39.68 -3 $EXTEND_RANGE_END 39.68 1 \
  --overwrite-cache

rm -r ../cache