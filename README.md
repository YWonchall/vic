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
ln -s /workspace/data/cooperative-vehicle-infrastructure /workspace/dair-v2x/data/DAIR-V2X
```
### 1.3 转换至kitti格式
```
# Kitti Format
cd /workspace/dair-v2x

python tools/dataset_converter/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure-example/infrastructure-side \
    --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure-example/infrastructure-side \
    --split-path ./data/split_datas/example-cooperative-split-data.json \
    --label-type lidar --sensor-view infrastructure --no-classmerge

python tools/dataset_converter/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure-example/vehicle-side \
    --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure-example/vehicle-side \
    --split-path ./data/split_datas/example-cooperative-split-data.json \
    --label-type lidar --sensor-view vehicle --no-classmerge
```
> 针对此比赛是不是应该merge？

### 1.4 jpg -> png
```
cd /workspace/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure-example/infrastructure-side/training/image_2/
for file in *.jpg; do mv $file ${file%%.*}.png; done

cd /workspace/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure-example/vehicle-side/training/image_2/
for file in *.jpg; do mv$file ${file%%.*}.png; done
```

### 1.5 create data
```
cd /workspace/mmdetection3d

python tools/create_data.py kitti --root-path \
/workspace/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure-example/infrastructure-side \
--out-dir \
/workspace/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure-example/infrastructure-side \
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
## 3. 更改测试或者推理数据集
- 更改数据目录
- 更改输出目录
- 更改划分json
- 更改split
- -- test

TODO:
- 提交github

