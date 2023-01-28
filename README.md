# 整体流程
## 1. 制作数据集
### 1.1 DAIR-V2X-C 数据集原始格式
```
# For DAIR-V2X-C Dataset located at ${DAIR-V2X-C_DATASET_ROOT}
└── cooperative-vehicle-infrastructure       <-- DAIR-V2X-C
    └──── infrastructure-side                         <-- DAIR-V2X-C-I   
       ├───── image
       ├───── velodyne
       ├───── calib
       ├───── label    
       └────  data_info.json    
    └──── vehicle-side                                         <-- DAIR-V2X-C-V  
       ├───── image
       ├───── velodyne
       ├───── calib
       ├───── label
       └───── data_info.json
    └──── cooperative 
       ├───── label_world
       └───── data_info.json              
```
### 1.2 建立数据集软链接
```
ln -s /workspace/vic-competition/data/cooperative-vehicle-infrastructure-test /workspace/vic-competition/dair-v2x/data/DAIR-V2X
```
### 1.3 转换至kitti格式
```
# Kitti Format
cd /workspace/vic-competition/dair-v2x

python tools/dataset_converter/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side \
    --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side \
    --split-path ./data/split_datas/cooperative-split-data.json \
    --label-type lidar --sensor-view infrastructure

python tools/dataset_converter/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side \
    --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side \
    --split-path ./data/split_datas/cooperative-split-data.json \
    --label-type lidar --sensor-view vehicle
```
> 针对此比赛是不是应该merge？

### 1.4 jpg -> png
```
cd /workspace/vic-competition/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side/training/image_2/
for file in *.jpg; do mv $file ${file%%.*}.png; done

cd /workspace/vic-competition/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/training/image_2/
for file in *.jpg; do mv $file ${file%%.*}.png; done
```

### 1.5 create data
```
cd /workspace/vic-competition/mmdetection3d

python tools/create_data.py kitti --root-path \
/workspace/vic-competition/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side \
--out-dir \
/workspace/vic-competition/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side \
--extra-tag kitti

python tools/create_data.py kitti --root-path \
/workspace/vic-competition/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side \
--out-dir \
/workspace/vic-competition/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side \
--extra-tag kitti
```

## 2. 数据集可视化
> 这里使用mmdetection里的可视化方法，dair-v2x中的可视化不可保存结果

可视化的数据分为图片和点云，分别对应于不同的task，源码如下：
```python
if vis_task in ['det', 'multi_modality-det']:
    # show 3D bboxes on 3D point clouds
    show_det_data(
        idx, dataset, args.output_dir, file_name, show=args.online)
if vis_task in ['multi_modality-det', 'mono-det']:
    # project 3D bboxes to 2D image
    show_proj_bbox_img(
        idx,
        dataset,
        args.output_dir,
        file_name,
        show=args.online,
        is_nus_mono=(dataset_type == 'NuScenesMonoDataset'))
 ```
 - det: 可视化点云
 - multi_modality-det: 可视化点云 + 将点云label投影至2d图片
 - mono-det：单目摄像头图片可视化

 可视化不同数据需要在配置文件中配置好加载方法，即文末的eval_pipeline。这里使用mvxnet多模态配置文件可以加载点云+图像。
 ```
 python tools/misc/browse_dataset.py /workspace/dair-v2x/configs/sv3d-veh/mvxnet/trainval_config.py --output-dir ./work-dirs/exam-c/inf/vis_dataset/ --task multi_modality-det
 ```

## 3. 训练
```
python tools/train.py ../dair-v2x/configs/vic3d/late-fusion-multimodal/mvxnet/trainval_config_i.py --work-dir /workspace/mmdetection3d/work-dirs/exam-c/inf/train
```
- config中的evaluation中的interval以epoch为单位，每多个epoch使用验证集评估一次
- log_config表示每迭代多少次打印一次信息，与迭代次数无关(由batchsize决定)

## 4. 可视化训练结果
```
python tools/analysis_tools/analyze_logs.py plot_curve  \
/workspace/vic-competition/mmdetection3d/work-dirs/vic/veh/train/20230126_211220.log.json \
--keys loss_cls loss_bbox \
--out losses.png
```

## 4. 单端推理/预测
```
sh scripts/eval.sh
```
- 使用eval及相关参数在验证集上做评估
- 使用format-only可以生成格式化的结果
- 可视化由于远程没有gui暂时不可用

## 5. 融合
> - 前融合需要匹配的数据对作为训练数据集，后融合无需车路端匹配，单独训练。
> - 推理时所有融合方法都需要匹配的数据对作为输入，且评估label用的是联合标注。
### 5.1 early fusion
方法：
- 从V2X-C中选出时间同步车路点云数据对，将车端和路端标注均转换至世界坐标系下，根据目标框匹配规则生成联合标注
- 将标注结果转换至车端lidar坐标系(替换车端原数据的标注)
- 将路端点云转换至车端lidar坐标系，并与车端融合
- 将联合标注数据的data_info.json替换车端data_info.json(仅用联合标注的数据对作为数据集)
> - 推理时同样进行了路端点云的转换(需要时间同步或异步的车路端数据对作为输入)
> - 经过上述融合后相当于变成了一个单端数据集
### 5.2 late fusion
- 单独训练车路端两个模型
- 推理时对两种数据单独推理
- 将推理结果均转换至世界坐标系进行匹配融合，然后转换至车端lidar坐标系输出。
```
bash scripts/infer_lidar_late_fusion_pointpillars.sh 0 late_fusion 2 0 100
```
> 时间异步的VICFrame没有数据
## 6. 提交
```
cd /workspace/vic-competition/dair-v2x/work-dirs/output/vic-late-multimodal-mvxnet-test/result/
zip ../test.zip ./*
```
# 注意点
## 1. mmdetection可视化数据集
- 不同配置文件读取的数据类型不同，只能可视化其对应类型的数据
- imvoxelnet的配置文件没有eval_pipeline，需要自己添加
    ```python
    eval_pipeline = [
        #dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
        dict(type="LoadImageFromFile"),
        dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
        dict(type="Collect3D", keys=["img"]),
    ]
    ```
## 2. 多模态模型如mvxnet，数据量翻倍(img+pointcloud)
## 3. 后融合更改测试或者推理数据集
- 更改数据目录
- 更改输出目录
- 更改划分json
- 更改split
- -- test
## 4. 训练log可视化
- 中断的训练不能可视化
- 可以采用覆盖的方式再次可视化
> 观察连续epoch的log文件的格式，可以自己合并

## 5.配置文件修改
- data_root
- lr
- lr_config
- max_epoch
- load_from
- batch_size

## 6. 目前仅修改了后融合支持mvxnet和custom评估

# TODO:
- 改进融合方法
    - match
    - fuse
        - diff<1 两个端的都很近，可考虑增加diff
range？

# 评估结果
> bev-0.5
## 1. 路端
| view | num_class|model|dataset | mAP| 
|-|-|-|-|-|
| inf | 3 | mvxnet_base| inf_val|35.6901|
| inf | 3 | mvxnet_my| inf_val | 44.8988 |
| inf | 3 | pointpillars_base| inf_val | 54.4083 |
| inf | 3 | pointpillars_my| inf_val | **62.7306** |

## 2.车端
| view | num_class|model|dataset | mAP| 
|-|-|-|-|-|
| veh | 3 | mvxnet_base| veh_val|70.0941|
| veh | 3 | mvxnet_my| veh_val | **72.1771** |
| veh | 3 | pointpillars_base| veh_val| 63.3766|
| veh | 3 | pointpillars_my| veh_val| 72.3666|
| veh | 1 | pointpillars_my| veh_val| 72.2730|


## 3.后融合base
| view | num_class|inf_model|veh_model|dataset | mAP| 
|-|-|-|-|-|-|
| late_fusion| 3 | mvxnet_base|mvxnet_base| cooperative_val | 49.91 |
|  late_fusionl| 3 | mvxnet_my|mvxnet_my| cooperative_val | 51.68 |
|  late_fusionl| 3 | pointpillars_base|pointpillars_base| cooperative_val | 62.01 |
|  late_fusionl| 3 | pointpillars_base|mvxnet_my| cooperative_val | 63.57 |
|  late_fusionl| 3 | pointpillars_my|mvxnet_my| cooperative_val |**65.20** |
|  late_fusionl| 3 | pointpillars_my|pointpillars_my| cooperative_val |64.40 |
|  late_fusionl| 3+1 | pointpillars_my|pointpillars_my| cooperative_val |30.11 |
|  late_fusionl| 3 | mvxnet_base|mvxnet_base| cooperative_test | 38.54241 |
|  late_fusionl| 3 | mvxnet_my|mvxnet_my| cooperative_test | 44.32577 |
|  late_fusionl| 3 | pointpillars_base|pointpillars_base| cooperative_test | 50.08663 |
|  late_fusionl| 3 | pointpillars_base|mvxnet_my| cooperative_test | 57.0815|
|  late_fusionl| 3 | pointpillars_my|mvxnet_my| cooperative_test | **58.2941**|

## 4.单车端
| view | num_class|model|dataset | mAP| 
|-|-|-|-|-|
| veh | 3 | pointpillars_my| cooperative_val| 67.24|
| veh | 3 | pointpillars_my| cooperative_test| 40.31732|

# backup
## 1.inf
### 1.1 mvxnet_base
```
    Pedestrian AP@0.50, 0.50, 0.50:
    bbox AP:28.2399, 29.8431, 29.7443
    bev  AP:25.9225, 26.8371, 26.4173
    3d   AP:23.3004, 23.7382, 23.7093
    aos  AP:23.26, 24.50, 24.45
    Pedestrian AP@0.50, 0.25, 0.25:
    bbox AP:28.2399, 29.8431, 29.7443
    bev  AP:33.6506, 34.1378, 34.0817
    3d   AP:33.6288, 34.0653, 34.0582
    aos  AP:23.26, 24.50, 24.45
    Cyclist AP@0.50, 0.50, 0.50:
    bbox AP:31.0266, 27.4884, 28.0700
    bev  AP:35.4790, 30.7848, 31.1174
    3d   AP:34.0262, 29.1576, 29.4814
    aos  AP:26.54, 23.82, 24.35
    Cyclist AP@0.50, 0.25, 0.25:
    bbox AP:31.0266, 27.4884, 28.0700
    bev  AP:36.8297, 31.8542, 32.1244
    3d   AP:36.8271, 31.8542, 32.1244
    aos  AP:26.54, 23.82, 24.35

    Car AP@0.70, 0.70, 0.70:
    bbox AP:10.4716, 8.5158, 8.6431
    *bev  AP:51.1279, 35.4104, 35.4209*
    3d   AP:47.1786, 33.5355, 33.4357
    aos  AP:10.23, 7.81, 7.92

    Car AP@0.70, 0.50, 0.50:
    bbox AP:10.4716, 8.5158, 8.6431
   * bev  AP:52.0527, 35.6901, 35.6953*
    3d   AP:51.8469, 35.6384, 35.6563
    aos  AP:10.23, 7.81, 7.92

    Overall AP@easy, moderate, hard:
    bbox AP:23.2460, 21.9491, 22.1525
    bev  AP:37.5098, 31.0108, 30.9852
    3d   AP:34.8351, 28.8104, 28.8755
    aos  AP:20.01, 18.71, 18.91
```
### 1.2 mvxnet_my
```
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:27.3328, 27.0255, 27.0206
bev  AP:20.2937, 19.8641, 19.7801
3d   AP:17.9189, 17.3346, 17.3117
aos  AP:13.24, 13.77, 13.78
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:27.3328, 27.0255, 27.0206
bev  AP:32.1801, 32.5594, 32.5637
3d   AP:31.9915, 32.3483, 32.3125
aos  AP:13.24, 13.77, 13.78
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:31.6355, 27.3483, 27.2990
bev  AP:34.8899, 27.9551, 27.7304
3d   AP:33.7100, 26.5496, 26.5334
aos  AP:14.96, 12.99, 12.97
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:31.6355, 27.3483, 27.2990
bev  AP:36.6405, 30.3792, 30.3403
3d   AP:36.5909, 30.3052, 30.2812
aos  AP:14.96, 12.99, 12.97

Car AP@0.70, 0.70, 0.70:
bbox AP:14.0273, 10.0283, 9.9222
*bev  AP:60.9007, 43.2108, 43.2550*
3d   AP:53.9171, 42.4701, 42.4956
aos  AP:7.04, 5.03, 4.98

Car AP@0.70, 0.50, 0.50:
bbox AP:14.0273, 10.0283, 9.9222
*bev  AP:62.0278, 44.8988, 44.8150*
3d   AP:61.2723, 44.7760, 44.5461
aos  AP:7.04, 5.03, 4.98

Overall AP@easy, moderate, hard:
bbox AP:24.3319, 21.4674, 21.4139
bev  AP:38.6947, 30.3434, 30.2552
3d   AP:35.1820, 28.7848, 28.7802
aos  AP:11.75, 10.60, 10.58
```
### 1.3 pointpillars_base
```
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:17.8167, 16.7477, 16.7722
bev  AP:16.4735, 14.8349, 14.8543
3d   AP:13.6064, 14.3551, 14.3167
aos  AP:8.60, 8.36, 8.38
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:17.8167, 16.7477, 16.7722
bev  AP:20.9548, 21.6055, 21.4750
3d   AP:20.9285, 21.5372, 21.4015
aos  AP:8.60, 8.36, 8.38
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:43.6380, 35.6700, 35.5814
bev  AP:47.2320, 37.6003, 37.3070
3d   AP:45.3614, 36.5875, 36.2582
aos  AP:21.89, 18.07, 18.03
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:43.6380, 35.6700, 35.5814
bev  AP:49.5507, 42.5385, 41.9823
3d   AP:49.5287, 42.3927, 41.9105
aos  AP:21.89, 18.07, 18.03
Car AP@0.70, 0.70, 0.70:
bbox AP:16.6960, 13.0129, 12.9518
bev  AP:71.9958, 54.0420, 54.0073
3d   AP:67.4906, 50.6824, 50.6367
aos  AP:8.94, 6.95, 6.91
Car AP@0.50, 0.50, 0.50:
bbox AP:63.9385, 49.6266, 49.7131
bev  AP:72.5277, 54.4083, 54.4032
3d   AP:72.3194, 54.2693, 54.2510
aos  AP:34.70, 26.89, 26.89

Overall AP@easy, moderate, hard:
bbox AP:26.0502, 21.8102, 21.7685
bev  AP:45.2337, 35.4924, 35.3895
3d   AP:42.1528, 33.8750, 33.7372
aos  AP:13.14, 11.13, 11.11
```
### 1.4 pointpillars_my
```
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:11.4714, 12.0390, 12.0903
bev  AP:13.2119, 14.3905, 14.3973
3d   AP:12.6036, 13.7621, 13.7149
aos  AP:5.10, 5.76, 5.80
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:11.4714, 12.0390, 12.0903
bev  AP:19.1944, 17.7737, 17.7992
3d   AP:19.1271, 17.8231, 17.8319
aos  AP:5.10, 5.76, 5.80
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:40.9792, 33.5943, 33.4847
bev  AP:44.6152, 35.0341, 34.8998
3d   AP:43.4100, 34.3125, 31.2970
aos  AP:20.61, 17.12, 17.05
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:40.9792, 33.5943, 33.4847
bev  AP:47.9038, 37.6506, 37.2466
3d   AP:47.8861, 37.6235, 37.1863
aos  AP:20.61, 17.12, 17.05
Car AP@0.70, 0.70, 0.70:
bbox AP:19.3466, 12.6745, 12.6402
bev  AP:80.7556, 62.1425, 54.1763
3d   AP:69.6249, 52.0501, 51.8715
aos  AP:10.39, 6.79, 6.77
Car AP@0.50, 0.50, 0.50:
bbox AP:64.3374, 49.5893, 49.6905
bev  AP:81.5873, 62.7306, 62.7452
3d   AP:81.3309, 62.5255, 62.5267
aos  AP:34.98, 26.86, 26.87

Overall AP@easy, moderate, hard:
bbox AP:23.9324, 19.4359, 19.4051
bev  AP:46.1943, 37.1891, 34.4911
3d   AP:41.8795, 33.3749, 32.2944
aos  AP:12.03, 9.89, 9.87
```
## 2.veh
### 2.1 mvxnet_pretrain
```
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:12.7357, 12.5691, 12.5127
bev  AP:50.8942, 40.9427, 40.5325
3d   AP:43.5775, 33.6627, 33.2673
aos  AP:3.91, 3.38, 3.38
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:12.7357, 12.5691, 12.5127
bev  AP:66.5516, 58.6684, 58.4038
3d   AP:66.1426, 58.5047, 58.2393
aos  AP:3.91, 3.38, 3.38
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:14.3683, 15.8634, 16.0248
bev  AP:45.8257, 43.4086, 40.0853
3d   AP:43.2608, 36.9887, 36.0007
aos  AP:4.05, 5.36, 5.74
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:14.3683, 15.8634, 16.0248
bev  AP:54.6427, 53.1212, 52.7382
3d   AP:54.6072, 52.9541, 49.1421
aos  AP:4.05, 5.36, 5.74
Car AP@0.70, 0.70, 0.70:
bbox AP:0.0000, 0.0027, 0.0035
bev  AP:68.6844, 62.2447, 61.4334
3d   AP:0.0008, 0.0014, 0.0014
aos  AP:0.00, 0.00, 0.00
Car AP@0.50, 0.50, 0.50:
bbox AP:0.0008, 0.0855, 0.1006
bev  AP:70.6463, 70.0941, 62.9587
3d   AP:0.0051, 0.0174, 0.0231
aos  AP:0.00, 0.06, 0.07

Overall AP@easy, moderate, hard:
bbox AP:9.0347, 9.4784, 9.5137
bev  AP:55.1348, 48.8653, 47.3504
3d   AP:28.9464, 23.5509, 23.0898
aos  AP:2.65, 2.92, 3.04
```
### 2.2 mvxnet_my
```
Results of pts_bbox:
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:11.3531, 11.0362, 11.0367
bev  AP:46.7263, 37.9860, 37.6561
3d   AP:40.2772, 31.9147, 31.2285
aos  AP:8.25, 8.00, 8.00
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:11.3531, 11.0362, 11.0367
bev  AP:60.0908, 49.2189, 48.7148
3d   AP:59.8402, 48.6301, 48.3766
aos  AP:8.25, 8.00, 8.00
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:15.6615, 14.3121, 14.3858
bev  AP:47.6318, 40.0764, 39.6152
3d   AP:45.8842, 36.4720, 35.8462
aos  AP:8.12, 7.43, 7.47
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:15.6615, 14.3121, 14.3858
bev  AP:51.1756, 43.4858, 43.0452
3d   AP:51.1688, 43.4137, 42.8906
aos  AP:8.12, 7.43, 7.47

Car AP@0.70, 0.70, 0.70:
bbox AP:19.1376, 17.0354, 16.6886
bev  AP:71.8279, 70.8212, 68.2867
3d   AP:69.7046, 61.1549, 59.5435
aos  AP:9.63, 8.56, 8.39
Car AP@0.50, 0.50, 0.50:
bbox AP:66.0179, 57.1045, 55.6848
bev  AP:78.9340, 72.1771, 71.8658
3d   AP:77.4589, 71.8710, 71.2644
aos  AP:33.17, 28.64, 27.93

Overall AP@easy, moderate, hard:
bbox AP:15.3841, 14.1279, 14.0370
bev  AP:55.3953, 49.6279, 48.5193
3d   AP:51.9553, 43.1806, 42.2060
aos  AP:8.66, 8.00, 7.95
```
### 2.3 pointpillars_base
```
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:10.8579, 10.3834, 10.3973
bev  AP:50.9905, 41.6415, 41.2759
3d   AP:47.4923, 35.5629, 35.2392
aos  AP:6.23, 6.03, 6.04
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:10.8579, 10.3834, 10.3973
bev  AP:61.2710, 50.7736, 50.0837
3d   AP:61.2530, 50.5773, 49.6543
aos  AP:6.23, 6.03, 6.04
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:13.6743, 15.0079, 15.1896
bev  AP:54.0369, 50.8888, 48.3082
3d   AP:52.7263, 47.5243, 46.3435
aos  AP:7.17, 7.82, 7.91
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:13.6743, 15.0079, 15.1896
bev  AP:56.5588, 54.3782, 53.2033
3d   AP:56.5479, 54.2831, 53.0851
aos  AP:7.17, 7.82, 7.91
Car AP@0.70, 0.70, 0.70:
bbox AP:15.5466, 17.9663, 17.5411
bev  AP:62.5964, 62.9226, 62.7879
3d   AP:59.9243, 58.6556, 52.3770
aos  AP:7.75, 8.94, 8.73
Car AP@0.50, 0.50, 0.50:
bbox AP:56.1863, 49.3003, 49.0349
bev  AP:69.9503, 63.3766, 63.3175
3d   AP:69.5043, 63.1803, 63.0868
aos  AP:27.96, 24.60, 24.47

Overall AP@easy, moderate, hard:
bbox AP:13.3596, 14.4525, 14.3760
bev  AP:55.8746, 51.8176, 50.7907
3d   AP:53.3810, 47.2476, 44.6532
aos  AP:7.05, 7.60, 7.56
```
### 2.4 pointpillars_my
```
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:10.5127, 10.3590, 10.3536
bev  AP:50.6387, 39.8114, 38.7553
3d   AP:45.3320, 34.9646, 34.4695
aos  AP:6.22, 6.16, 6.16
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:10.5127, 10.3590, 10.3536
bev  AP:60.2321, 48.8070, 48.0828
3d   AP:60.0005, 48.3322, 47.7990
aos  AP:6.22, 6.16, 6.16
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:9.6701, 10.5040, 10.6846
bev  AP:53.0705, 48.9516, 48.3916
3d   AP:51.9410, 47.1727, 45.4976
aos  AP:4.81, 5.44, 5.52
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:9.6701, 10.5040, 10.6846
bev  AP:56.0501, 54.2094, 52.8188
3d   AP:56.0317, 53.9990, 52.5183
aos  AP:4.81, 5.44, 5.52
Car AP@0.70, 0.70, 0.70:
bbox AP:23.3892, 21.0604, 18.0361
bev  AP:71.7593, 71.6770, 70.2539
3d   AP:69.2207, 61.1673, 60.6133
aos  AP:11.57, 10.61, 9.10

Car AP@0.50, 0.50, 0.50:
bbox AP:64.8585, 56.7568, 55.7434
bev  AP:79.0132, 72.3666, 72.2549
3d   AP:78.6647, 72.1014, 71.9022
aos  AP:32.35, 28.46, 27.96

Overall AP@easy, moderate, hard:
bbox AP:14.5240, 13.9745, 13.0248
bev  AP:58.4895, 53.4800, 52.4669
3d   AP:55.4979, 47.7682, 46.8601
aos  AP:7.54, 7.40, 6.93
```
## 3.late_fusion
### 3.1 mvxnet_pretrained
```
    car 3d IoU threshold 0.30, Average Precision = 5.01
    car 3d IoU threshold 0.50, Average Precision = 0.92
    car 3d IoU threshold 0.70, Average Precision = 0.49
    car bev IoU threshold 0.30, Average Precision = 57.17
    car bev IoU threshold 0.50, Average Precision = 49.91
    car bev IoU threshold 0.70, Average Precision = 41.43
    Average Communication Cost = 602.48 Bytes
```
### 3.2 mvxnet_my_inf
```
    car 3d IoU threshold 0.30, Average Precision = 7.14
    car 3d IoU threshold 0.50, Average Precision = 1.91
    car 3d IoU threshold 0.70, Average Precision = 1.13
    car bev IoU threshold 0.30, Average Precision = 59.84
    car bev IoU threshold 0.50, Average Precision = 51.78
    car bev IoU threshold 0.70, Average Precision = 42.60
```
### 3.3 mvxnet_my
```
car 3d IoU threshold 0.30, Average Precision = 53.95
car 3d IoU threshold 0.50, Average Precision = 48.87
car 3d IoU threshold 0.70, Average Precision = 36.65
car bev IoU threshold 0.30, Average Precision = 54.83
car bev IoU threshold 0.50, Average Precision = 51.68
car bev IoU threshold 0.70, Average Precision = 46.20
Average Communication Cost = 274.24 Bytes
```
### 3.4 pointpillars_base
```
car 3d IoU threshold 0.30, Average Precision = 65.56
car 3d IoU threshold 0.50, Average Precision = 55.98
car 3d IoU threshold 0.70, Average Precision = 40.02
car bev IoU threshold 0.30, Average Precision = 67.40
car bev IoU threshold 0.50, Average Precision = 62.01
car bev IoU threshold 0.70, Average Precision = 54.12
Average Communication Cost = 478.52 Bytes
```
### 3.5 pointpillars_base + mvxnet_my
```
car 3d IoU threshold 0.30, Average Precision = 66.75
car 3d IoU threshold 0.50, Average Precision = 58.44
car 3d IoU threshold 0.70, Average Precision = 42.45
car bev IoU threshold 0.30, Average Precision = 68.55
car bev IoU threshold 0.50, Average Precision = 63.57
car bev IoU threshold 0.70, Average Precision = 55.26
Average Communication Cost = 478.52 Bytes
```
### 3.6 pointpillars_my + mvxnet_my
```
car 3d IoU threshold 0.30, Average Precision = 68.57
car 3d IoU threshold 0.50, Average Precision = 59.88
car 3d IoU threshold 0.70, Average Precision = 43.24
car bev IoU threshold 0.30, Average Precision = 70.43
car bev IoU threshold 0.50, Average Precision = 65.20
car bev IoU threshold 0.70, Average Precision = 56.54
Average Communication Cost = 491.60 Bytes
```
### 3.7 pointpillars_my
```
car 3d IoU threshold 0.30, Average Precision = 67.89
car 3d IoU threshold 0.50, Average Precision = 58.80
car 3d IoU threshold 0.70, Average Precision = 43.71
car bev IoU threshold 0.30, Average Precision = 69.79
car bev IoU threshold 0.50, Average Precision = 64.40
car bev IoU threshold 0.70, Average Precision = 56.63
Average Communication Cost = 517.20 Bytes
```
## 4. veh_only
### 4.1 pointpillars_my
```
car 3d IoU threshold 0.30, Average Precision = 67.49
car 3d IoU threshold 0.50, Average Precision = 65.50
car 3d IoU threshold 0.70, Average Precision = 52.35
car bev IoU threshold 0.30, Average Precision = 67.71
car bev IoU threshold 0.50, Average Precision = 67.24
car bev IoU threshold 0.70, Average Precision = 64.46
```