# 整体流程
## 1. 制作数据集
### 1.1制作后融合数据集
1.1.1 DAIR-V2X-C 数据集原始格式
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
1.1.2 建立数据集软链接
```
ln -s /workspace/vic-competition/data/cooperative-vehicle-infrastructure-test /workspace/vic-competition/dair-v2x/data/DAIR-V2X
```

1.1.3 转换至kitti格式
> 可选：class merge
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


1.1.4 jpg -> png
```
cd /workspace/vic-competition/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side/training/image_2/
for file in *.jpg; do mv $file ${file%%.*}.png; done

cd /workspace/vic-competition/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/training/image_2/
for file in *.jpg; do mv $file ${file%%.*}.png; done
```

1.1.5 create data
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

### 1.2. 制作前融合数据集
1.2.1 准备目录
```
cp -r data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training
```

1.2.2 将联合标注转至车端lidar
```
rm -r ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/label/lidar

python tools/dataset_converter/label_world2v.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/label/lidar
```

1.2.3 转换路端点云至车端
```
python tools/dataset_converter/point_cloud_i2v.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne/lidar_i2v --num-worker 16
```

1.2.4 融合点云
```
python tools/dataset_converter/concatenate_pcd2bin.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --i2v-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne/lidar_i2v --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne-concated

rm -r ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne

mv ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne-concated  ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne
```

1.2.5 转换信息文件
```
python tools/dataset_converter/get_fusion_data_info.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure  --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training

rm ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/data_info.json

mv ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/fusion_data_info.json ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/data_info.json
```

1.2.6 创建kitti数据集
```
python tools/dataset_converter/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training --split-path ./data/split_datas/cooperative-split-data.json --label-type lidar --sensor-view cooperative
```

1.2.7 jpg -> png
```
cd /workspace/vic-competition/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/training/image_2/
for file in *.jpg; do mv $file ${file%%.*}.png; done
```

1.2.8 create data
```
cd /workspace/vic-competition/mmdetection3d

python tools/create_data.py kitti --root-path \
/workspace/vic-competition/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training \
--out-dir \
/workspace/vic-competition/dair-v2x/data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training \
--extra-tag kitti
```
1783

## 2. 数据集可视化
> - 这里使用mmdetection里的可视化方法，dair-v2x中的可视化不可保存结果
> - 可视化的数据分为图片和点云，分别对应于不同的task，源码如下：
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
#  det: 可视化点云
#  multi_modality-det: 可视化点云 + 将点云label投影至2d图片
#  mono-det：单目摄像头图片可视化

 ```
 
 > 可视化不同数据需要在配置文件中配置好加载方法，即文末的eval_pipeline。这里使用mvxnet多模态配置文件可以加载点云+图像。
 ```
 python tools/misc/browse_dataset.py /workspace/vic-competition/dair-v2x/configs/sv3d-veh/mvxnet/trainval_config.py --output-dir ./work-dirs/vic/veh/early-fusion/vis --task multi_modality-det
 ```

## 3. 训练
```
python tools/train.py ../dair-v2x/configs/vic3d/late-fusion-multimodal/mvxnet/trainval_config_i.py --work-dir /workspace/mmdetection3d/work-dirs/exam-c/inf/train
```
> - onfig中的evaluation中的interval以epoch为单位，每多个epoch使用验证集评估一次
> - log_config表示每迭代多少次打印一次信息，与迭代次数无关(由batchsize决定)
> - 迭代次数 = num_examples/batch_size

## 4. 可视化训练log
```
python tools/analysis_tools/analyze_logs.py plot_curve  \
/workspace/vic-competition/mmdetection3d/work-dirs/vic/veh/train/20230126_211220.log.json \
--keys loss_cls loss_bbox \
--out losses.png
```

## 5. 单端推理/预测
```
sh scripts/eval.sh
```

> - 使用eval及相关参数在验证集上做评估
> - 使用format-only可以生成格式化的结果
> - 可视化由于远程没有gui暂时不可用

## 7. 融合
> - 训练：前融合需要匹配的数据对作为训练数据集，后融合无需车路端匹配，单独训练。
> - 推理时所有融合方法都需要匹配的数据对作为输入，且评估label用的是联合标注。
### 7.1 early fusion
方法：
    - 从V2X-C中选出时间同步车路点云数据对，将车端和路端标注均转换至世界坐标系下，根据目标框匹配规则生成联合标注
    - 将标注结果转换至车端lidar坐标系(替换车端原数据的标注)
    - 将路端点云转换至车端lidar坐标系，并与车端融合
    - 将联合标注数据的data_info.json替换车端data_info.json(仅用联合标注的数据对作为数据集)
> - 推理时同样进行了路端点云的转换(需要时间同步或异步的车路端数据对作为输入)
> - 经过上述融合后相当于变成了一个单端数据集
> - 训练离线融合，推理测试在线融合
```
bash scripts/eval_lidar_early_fusion_pointpillars.sh 0 early_fusion 1 0 100
```
> early_fusion 数据集和其他的相同，只不过在模型推理时进行了点云融合，转换kitti数据集仅为了训练
### 7.2 late fusion
> - 单独训练车路端两个模型
> - 推理时对两种数据单独推理(数据对)
> - 将推理结果均转换至世界坐标系进行匹配融合，然后转换至车端lidar坐标系输出。
```
bash scripts/infer_lidar_late_fusion_pointpillars.sh 0 late_fusion 2 0 100
```


## 8. 提交
```
cd /workspace/vic-competition/dair-v2x/work-dirs/output/vic-late-multimodal-mvxnet-test/result/
zip ../test.zip ./*
```

# 注意点
## 1. 训练
2. 多模态模型如mvxnet，数据量翻倍(img+pointcloud)
3. 配置文件修改
    - data_root
    - lr
    - lr_config
    - max_epoch
    - load_from
    - batch_size
## 2. 可视化
1. mmdetection可视化数据集
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
2. 训练log可视化
    - 中断的训练不能可视化
    - 可以采用覆盖的方式再次可视化
    > 观察连续epoch的log文件的格式，可以自己合并

## 3.推理测试
1. 后融合更改测试或者推理数据集
    - 更改数据目录
    - 更改输出目录
    - 更改划分json
    - 更改split
    - -- test



## 4. 标注
1. 联合标注里只有四类车，单端各类均有
2. 预测出的box非长方体，不规则，但是作图要求长方体，因此图与实际预测有出入(论文考虑)

## 5.融合
1. 对于单车类模型，其预测label为0，而数据集中label为2，需要对预测结果强制转换(model脚本)
    - late_fusion 正常映射？
    - early_fusion 强制置2
    - single_only 未处理

## 6. 命名方式
- 模型： 模型名称_类别数_所属端侧_数据集名称_(base).pth
- 数据集：数据集名称_所属端侧_所属划分
- config文件：trainval_config_端_类.py
- config目录：融合方法/模态/模型
    - early-fusion
    - filted-early-fusion
    - late-fusion

# TODO:
> TODO 时间异步的VICFrame没有数据
## 1.后融合
    - 改进融合方法
        - match
        - fuse
            - diff<1 两个端的都很近，可考虑增加diff
## 2.前融合
1. 过滤方法
    - score过滤
    - 过滤范围
    - 中心点选取
2. **模型精度比较**
    
range？


# 评估结果
> - bev-0.5-modest 
## 1. 单端
### 1.1 车端
| view | num_class|model|eval_dataset | mAP| 
|-|-|-|-|-|
| veh | 3 | mvxnet_veh_3_vic_veh_base.pth| vic_ veh_val |70.0941|
| veh | 3 | mvxnet_veh_3_vic_veh.pth| vic_ veh_val | 72.1771 |
| veh | 3 | pointpillars_veh_3_vic_veh_base.pth| vic_veh_val| 63.3766|
| veh | 3 | pointpillars_veh_3_vic_veh.pth| vic_veh_val| 72.3666|
| veh | 1 | pointpillars_veh_1_vic_veh.pth| vic_veh_val| 72.2730|

### 1.2 路端
| view | num_class|model|eval_dataset | mAP| 
|-|-|-|-|-|
| inf | 3 | mvxnet_inf_3_vic_inf_base.pth| vic_inf_val|35.6901|
| inf | 3 | mvxnet_inf_3_vic_inf.pth| vic_inf_val | 44.8988 |
| inf | 3 | pointpillars_inf_3_vic_inf_base.pth| vic_inf_val | 54.4083 |
| inf | 3 | pointpillars_inf_3_vic_inf.pth| vic_inf_val | 62.7306 |

## 2. 后融合

| fusion_method |inf_model|veh_model|eval_dataset | mAP| 
|-|-|-|-|-|
| late_fusion| mvxnet_inf_3_sv_inf_base.pth |mvxnet_veh_3_sv_veh_base.pth| vic_coop_val | 49.91 |
|  late_fusion| mvxnet_inf_3_vic_inf | mvxnet_veh_3_vic_veh | vic_coop_val | 51.68 |
|  late_fusion| pointpillars_inf_3_vic_inf_base.pth |pointpillars_veh_3_vic_veh_base.pth| vic_coop_val | 62.01 |
|  late_fusiol| pointpillars_inf_3_vic_inf_base.pth| mvxnet_veh_3_vic_veh| vic_coop_val | 63.57 |
|  late_fusion|  pointpillars_inf_3_vic_inf|mvxnet_veh_3_vic_veh| vic_coop_val |**65.20** |
|  late_fusion|  pointpillars_inf_3_vic_inf|pointpillars_veh_3_vic_veh.pth| vic_coop_val |64.40 |
|  late_fusion|  pointpillars_inf_3_vic_inf|pointpillars_veh_1_vic_veh.pth| vic_coop_val |30.11 |
| late_fusion| mvxnet_inf_3_vic_inf_base.pth |mvxnet_veh_3_vic_veh_base.pth| vic_coop_test | 38.54241 |
|  late_fusion| mvxnet_inf_3_vic_inf | mvxnet_veh_3_vic_veh | vic_coop_test | 44.32577 |
|  late_fusiol| pointpillars_inf_3_vic_inf_base.pth |pointpillars_veh_3_vic_veh_base.pth| vic_coop_test | 50.08663 |
|  late_fusion| pointpillars_inf_3_vic_inf_base.pth| mvxnet_veh_3_vic_veh| vic_coop_test | 57.0815|
|  late_fusion|  pointpillars_inf_3_vic_inf|mvxnet_veh_3_vic_veh| vic_coop_test | **58.2941**|


## 3.前融合

| fusion_method | model|eval_dataset | mAP| ab_cost |
|-|-|-|-|-|
|veh_only|pointpillars_veh_1_vic_coop_base.pth|vic_coop_val_15|56.52|0|
|veh_only|mvxnet_veh_3_vic_veh.pth|vic_coop_val_15|50.35|0|
|early_fusion|pointpillars_veh_1_vic_coop_base.pth|vic_coop_val_15|67.75|2374|
|early_fusion|mvxnet_veh_3_vic_veh.pth|vic_coop_val_15|48.13|3497|
|early_fusion|pointpillars_veh_1_vic_coop_base.pth|vic_coop_test | 61.80672 | 2000|

| fusion_method |inf_model|veh_model|eval_dataset | mAP| ab_cost |ratio|
|-|-|-|-|-|-|-|
|filted_early_fusion|pointpillars_inf_3_vic_inf.pth|pointpillars_veh_3_vic_veh.pth|vic_coop_va_15|57.09|-|1|
|filted_early_fusion|pointpillars_inf_3_vic_inf.pth|pointpillars_veh_3_vic_veh.pth|vic_coop_va_15|55.53|-|2|
|filted_early_fusion|pointpillars_inf_3_vic_inf.pth|pointpillars_veh_1_vic_coop_base.pth|vic_coop_val_15|65.72|-|1|
|filted_early_fusion|pointpillars_inf_3_vic_inf.pth|pointpillars_veh_1_vic_coop_base.pth|vic_coop_va_15|-|-|1|







# 5

pypcd读取的点云，使用open3d保存成pcd后 与bro生成的obj不同方向
- obj pcd不同
- 读取时的源数据不同

可视化生成的obj坐标相同
模型输出的box和源点云相同
划分：
- 单车端train
- 单路端test testA
- 联合 train 删除？

单路端数据集中没有
联合标注有

相当于多创造了一个路端没有的样例

raceback (most recent call last):
  File "tools/create_data.py", line 222, in <module>
    out_dir=args.out_dir)
  File "tools/create_data.py", line 33, in kitti_data_prep
    kitti.export_2d_annotation(root_path, info_val_path)
  File "/workspace/vic-competition/mmdetection3d/tools/data_converter/kitti_converter.py", line 350, in export_2d_annotation
    coco_infos = get_2d_boxes(info, occluded=[0, 1, 2, 3], mono3d=mono3d)
  File "/workspace/vic-competition/mmdetection3d/tools/data_converter/kitti_converter.py", line 452, in get_2d_boxes
    final_coords = post_process_coords(corner_coords)
  File "/workspace/vic-competition/mmdetection3d/tools/data_converter/nuscenes_converter.py", line 551, in post_process_coords
    [coord for coord in img_intersection.exterior.coords])
AttributeError: 'LineString' object has no attribute 'exterior'

014338 样本有错误，从imagesets中剔除
063315路端没有 从联合标注中剔除

前融合训练集 由所有联合标注组成，根据split里的train val划分为训练验证

训练时用的验证为路端传输的所有点云，评估测试时可以自定义过滤方法，因此二者有一定差异，使用评估指标，训练时的验证指标仅作参考

问题： 如果自己训练前融合模型，训练集和 评估的验证集不同，是否会影响效果？

验证集可视化有问题