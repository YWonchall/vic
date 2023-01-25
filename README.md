# 整体流程(example)
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
ln -s /workspace/vic-competition/data/cooperative-vehicle-infrastructure /workspace/vic-competition/dair-v2x/data/DAIR-V2X
```
### 1.3 转换至kitti格式
```
# Kitti Format
cd /workspace/vic-competition/dair-v2x

python tools/dataset_converter/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side \
    --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side \
    --split-path ./data/split_datas/cooperative-split-data.json \
    --label-type lidar --sensor-view infrastructure

python tools/dataset_converter/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure-example/vehicle-side \
    --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure-example/vehicle-side \
    --split-path ./data/split_datas/example-cooperative-split-data.json \
    --label-type lidar --sensor-view vehicle
```
> 针对此比赛是不是应该merge？

### 1.4 jpg -> png
```
cd /workspace/vic-competition/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side/training/image_2/
for file in *.jpg; do mv $file ${file%%.*}.png; done

cd /workspace/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure-example/vehicle-side/training/image_2/
for file in *.jpg; do mv$file ${file%%.*}.png; done
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
/workspace/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure-example/vehicle-side \
--out-dir \
/workspace/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure-example/vehicle-side \
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
/workspace/vic-competition/mmdetection3d/work-dirs/vic/inf/train/20230123_214854.log.json \
--keys loss_cls loss_bbox \
--out losses.png
```

## 4. 单端推理/预测
```
sh scripts/eval_multimodal_single_mvxnet.sh
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
cd /workspace/dair-v2x/work-dirs/output/vic-late-multimodal-mvxnet-test/result/
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
# TODO:
> 车端未合并label，先看路端效果
- 比较不同模型精度
- 使用2020预训练模型重新训练一个

1.寻找可用预训练模型
- inf:
    - 比赛提供的不可用
    - 2021不可用
    - 2020可用



# 评估结果
## 1.vic_inf
### 1.1. mvxnet_pretrained
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
## 2.late_fusion
### 2.1 mvxnet_pretrained
```
    car 3d IoU threshold 0.30, Average Precision = 5.01
    car 3d IoU threshold 0.50, Average Precision = 0.92
    car 3d IoU threshold 0.70, Average Precision = 0.49
    car bev IoU threshold 0.30, Average Precision = 57.17
    car bev IoU threshold 0.50, Average Precision = 49.91
    car bev IoU threshold 0.70, Average Precision = 41.43
    Average Communication Cost = 602.48 Bytes
```
### 2.2 mvxnet_my_inf
```
    car 3d IoU threshold 0.30, Average Precision = 7.14
    car 3d IoU threshold 0.50, Average Precision = 1.91
    car 3d IoU threshold 0.70, Average Precision = 1.13
    car bev IoU threshold 0.30, Average Precision = 59.84
    car bev IoU threshold 0.50, Average Precision = 51.78
    car bev IoU threshold 0.70, Average Precision = 42.60
```
