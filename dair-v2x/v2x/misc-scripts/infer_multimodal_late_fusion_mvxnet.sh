DATA="../data/DAIR-V2X/cooperative-vehicle-infrastructure-test"
OUTPUT="../work-dirs/output/vic-late-multimodal-mvxnet-test"
rm -r $OUTPUT
mkdir -p $OUTPUT/result
mkdir -p $OUTPUT/inf/multimodal
mkdir -p $OUTPUT/veh/multimodal

INFRA_MODEL_PATH="../configs/vic3d/late-fusion-multimodal/mvxnet"
INFRA_CONFIG_NAME="trainval_config_i.py"
INFRA_MODEL_NAME="mvxnet_inf_my.pth" #"sv3d_inf_mvxnet_c2271983b04b73e573486fcbc559c31e.pth"

VEHICLE_MODEL_PATH="../configs/vic3d/late-fusion-multimodal/mvxnet"
VEHICLE_CONFIG_NAME="trainval_config_v.py"
VEHICLE_MODEL_NAME="mvxnet_veh_my.pth"

SPLIT_DATA_PATH="../data/split_datas/cooperative-split-data.json"

# srun --gres=gpu:a100:1 --time=1-0:0:0 --job-name "dair-v2x" \
CUDA_VISIBLE_DEVICES=0 #$1
FUSION_METHOD='late_fusion' #$2
DELAY_K=2 #$3
EXTEND_RANGE_START=0 #$4
EXTEND_RANGE_END=100 #$5
TIME_COMPENSATION=$6
python infer.py \
  --input $DATA \
  --output $OUTPUT \
  --model $FUSION_METHOD \
  --dataset vic-sync \
  --k $DELAY_K \
  --split test \
  --test \
  --split-data-path $SPLIT_DATA_PATH \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/$INFRA_MODEL_NAME \
  --veh-config-path $VEHICLE_MODEL_PATH/$VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH/${VEHICLE_MODEL_NAME} \
  --device ${CUDA_VISIBLE_DEVICES} \
  --pred-class car \
  --sensortype multimodal \
  --extended-range $EXTEND_RANGE_START -39.68 -3 $EXTEND_RANGE_END 39.68 1 \
  --overwrite-cache \
  $TIME_COMPENSATION