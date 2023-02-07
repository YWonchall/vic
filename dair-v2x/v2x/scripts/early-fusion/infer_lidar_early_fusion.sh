DATA="../data/DAIR-V2X/cooperative-vehicle-infrastructure-test"
OUTPUT="../work-dirs/output/vic-early-lidar-pointpillars-test"
rm -r ../cache
rm -r $OUTPUT

INFRA_MODEL_PATH="../configs/vic3d/filted-early-fusion/pointcloud/pointpillars"
INFRA_CONFIG_NAME="trainval_config_inf_3.py"
INFRA_MODEL_NAME="pointpillars_inf_3_vic_inf.pth" #"sv3d_inf_mvxnet_c2271983b04b73e573486fcbc559c31e.pth"
# 3
VEHICLE_MODEL_PATH="../configs/vic3d/filted-early-fusion/pointcloud/pointpillars"
VEHICLE_CONFIG_NAME="trainval_config_veh_1.py"
VEHICLE_MODEL_NAME="pointpillars_veh_1_vic_coop.pth"
# 4
SPLIT_DATA_PATH="../data/split_datas/cooperative-split-data.json"

# srun --gres=gpu:a100:1 --time=1-0:0:0 --job-name "dair-v2x" \
CUDA_VISIBLE_DEVICES=0 #$1
FUSION_METHOD='early_fusion' #$2
DELAY_K=1 #$3
EXTEND_RANGE_START=0 #$4
EXTEND_RANGE_END=100 #$5

# 5 sensortype
python infer.py \
  --input $DATA \
  --output $OUTPUT \
  --model $FUSION_METHOD \
  --dataset vic-sync \
  --k $DELAY_K \
  --set-label \
  --split test \
  --test \
  --split-data-path $SPLIT_DATA_PATH \
  --inf-filter \
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
  --overwrite-cache