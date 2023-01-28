DATA="../data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side"
# 1
OUTPUT="../work-dirs/output/vic-late-custom"
rm -r $OUTPUT
mkdir -p $OUTPUT/result
# 2
mkdir -p $OUTPUT/inf/lidar
mkdir -p $OUTPUT/veh/lidar
# 3
INFRA_MODEL_PATH="../configs/vic3d/late-fusion-pointcloud/pointpillars"
INFRA_CONFIG_NAME="trainval_config_i.py"
INFRA_MODEL_NAME="pointpillars_inf_my.pth" #"sv3d_inf_mvxnet_c2271983b04b73e573486fcbc559c31e.pth"
# 4
VEHICLE_MODEL_PATH="../configs/vic3d/late-fusion-pointcloud/pointpillars"
VEHICLE_CONFIG_NAME="trainval_config_v.py"
VEHICLE_MODEL_NAME="pointpillars_veh_my.pth"

SPLIT_DATA_PATH="../data/split_datas/cooperative-split-data.json"

# srun --gres=gpu:a100:1 --time=1-0:0:0 --job-name "dair-v2x" \
CUDA_VISIBLE_DEVICES=0 #$1
FUSION_METHOD='single_side' #$2
DELAY_K=2 #$3
EXTEND_RANGE_START=0 #$4
EXTEND_RANGE_END=100 #$5
TIME_COMPENSATION=$6
python eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model $FUSION_METHOD \
  --eval-single \
  --dataset dair-v2x-v \
  --k $DELAY_K \
  --split val \
  --split-data-path $SPLIT_DATA_PATH \
  --config-path $VEHICLE_MODEL_PATH/$VEHICLE_CONFIG_NAME \
  --model-path $VEHICLE_MODEL_PATH/${VEHICLE_MODEL_NAME} \
  --device ${CUDA_VISIBLE_DEVICES} \
  --pred-class car \
  --sensortype lidar \
  # 5
  --inf-sensortype lidar \
  --veh-sensortype lidar \
  --extended-range $EXTEND_RANGE_START -39.68 -3 $EXTEND_RANGE_END 39.68 1 \
  --overwrite-cache \
  $TIME_COMPENSATION