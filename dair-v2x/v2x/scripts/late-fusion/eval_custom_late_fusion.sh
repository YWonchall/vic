DATA="../data/DAIR-V2X/cooperative-vehicle-infrastructure"
# 1
OUTPUT="../work-dirs/output/vic-late-custom-pointpillars"
rm -r $OUTPUT
mkdir -p $OUTPUT/result
# 2
mkdir -p $OUTPUT/inf/lidar
mkdir -p $OUTPUT/veh/lidar
# 3
INFRA_MODEL_PATH="../configs/vic3d-report/late-fusion/pointpillars"
INFRA_CONFIG_NAME="trainval_config_inf_1.py"
INFRA_MODEL_NAME="pointpillars_inf.pth"
# 4
VEHICLE_MODEL_PATH="../configs/vic3d-report/late-fusion/pointpillars"
VEHICLE_CONFIG_NAME="trainval_config_veh_1.py"
VEHICLE_MODEL_NAME="pointpillars_veh.pth"

SPLIT_DATA_PATH="../data/split_datas/cooperative-split-data.json"

# srun --gres=gpu:a100:1 --time=1-0:0:0 --job-name "dair-v2x" \
CUDA_VISIBLE_DEVICES=0 #$1
FUSION_METHOD='late_fusion' #$2
DELAY_K=2 #$3
EXTEND_RANGE_START=0 #$4
EXTEND_RANGE_END=100 #$5
TIME_COMPENSATION=$6
# 5.sensortype
python eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model $FUSION_METHOD \
  --dataset vic-sync \
  --k $DELAY_K \
  --split val \
  --split-data-path $SPLIT_DATA_PATH \
  --set-inf-label \
  --set-veh-label \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/$INFRA_MODEL_NAME \
  --veh-config-path $VEHICLE_MODEL_PATH/$VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH/${VEHICLE_MODEL_NAME} \
  --device ${CUDA_VISIBLE_DEVICES} \
  --pred-class car \
  --sensortype lidar \
  --inf-sensortype lidar \
  --veh-sensortype lidar \
  --extended-range $EXTEND_RANGE_START -39.68 -3 $EXTEND_RANGE_END 39.68 1 \
  --overwrite-cache \
  $TIME_COMPENSATION