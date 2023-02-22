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
 python tools/misc/browse_dataset.py /workspace/vic-competition/dair-v2x/configs/sv3d-veh/mvxnet/trainval_config.py --output-dir ./work-dirs/vic/veh-coop-all/vis --task multi_modality-det
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
/workspace/inf_loss.json \
--keys loss_cls loss_bbox  loss \
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
3. 联合标注在世界坐标下，数据集加载时已转换至车端lidar坐标下

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
- 修改pipe ———— 需要改回
- 修改了传输点云的计算方式，若需要传输目标框，则改回
- mvxnet和second是不是没有训练彻底？改一下类别数目？用coop数据集继续训练
- 增加路端过滤模型
    - 使用second(使用sv-inf预训练)
- 测试benchmark中的模型和使用其预训练模型训练
- **车端过滤单类需要单独置2**
- sv-inf second模型精度挺高，考虑用其作为预训练模型重新训练一个路端 3类
- 确定好路 车(融合)的模型后 组合验证


# Question:
- 前融合训练集，训练时的val的map很低，ealy fusion eval时的map正常。eval是否将所有点云全部传输？
    - 过滤后的传输仍然很低
- mvxnet单类loss=0
- range?

- coop_val比vic_val低的原因：
    - coop_val使用联合标注的label,vic_val使用单端label（）
    - coop_val全部转车端lidar坐标评估，vic_val在各自端评估（影响不大）
- 总体而言，联合标注label跟任何一个单端都不匹配，但这不影响使用inf过滤，inf只需正确检测到目标点云然后保留即可，与联合标注label无关。但车端最佳方案为使用**过滤后的融合点云**，和联合标注的label训练。
- 使用vic-veh训练的模型与coop融合数据集训练的模型不仅在于点云多少，重要的是label不同，coop融合的label为评估label，更重要
- 路端 准 即可，只需保证推理和训练同分布，即训练集制作时的过滤依据的label与路端模型训练的label是相同的
- 车端 需要拟合label

# Idea
- 根据score决定过滤范围
- 预测的box非长方体，中心点的确定方法可能有一定影响
- **重新制作前融合训练集(使用过滤后的点融合)**
- 改进路端精度，参考那个比赛

# 评估结果 (bev-0.5-car-modest) 

## 前融合更换：
1. 数据集
    - 数据集root
    - split-data
    - %15
2. 模型
    - 模型路径
    - 传感器类型
    - 是否是单类-eval.py
3. 融合
    - 是否过滤


## 1. 单端(vic-veh/inf 验证集使用mmdetection3d中的test测试)
### 1.1 车端
| view | num_class|model|eval_dataset | mAP| 
|-|-|-|-|-|
| veh | 3 | mvxnet_veh_3_sv_veh_base.pth| vic_ veh_val |70.0941|
| veh | 3 | mvxnet_veh_3_vic_veh.pth| vic_ veh_val | 72.1771 |
| veh | 3 | pointpillars_veh_3_vic_veh_base.pth| vic_veh_val| 63.3766|
| veh | 3 | pointpillars_veh_3_vic_veh.pth| vic_veh_val| 72.3666|
| veh | 1 | pointpillars_veh_1_vic_veh.pth| vic_veh_val| 72.2730|
| veh | 3 | second_veh_3_sv_veh_base.pth| vic_veh_val| 62.5588|

### 1.2 路端
| view | num_class|model|eval_dataset | mAP| 
|-|-|-|-|-|
| inf | 3 | mvxnet_inf_3_sv_inf_base.pth| vic_inf_val|35.6901|
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
| late_fusion| mvxnet_inf_3_sv_inf_base.pth |mvxnet_veh_3_sv_veh_base.pth| vic_coop_test | 38.54241 |
|  late_fusion| mvxnet_inf_3_vic_inf | mvxnet_veh_3_vic_veh | vic_coop_test | 44.32577 |
|  late_fusiol| pointpillars_inf_3_vic_inf_base.pth |pointpillars_veh_3_vic_veh_base.pth| vic_coop_test | 50.08663 |
|  late_fusion| pointpillars_inf_3_vic_inf_base.pth| mvxnet_veh_3_vic_veh| vic_coop_test | 57.0815|
|  late_fusion|  pointpillars_inf_3_vic_inf|mvxnet_veh_3_vic_veh| vic_coop_test | **58.2941**|


## 3.前融合(vic-coop-veh 使用no fusion veh_only验证)

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


# 记录
1. pypcd读取的点云，使用open3d保存成pcd后 与bro生成的obj不同方向
- 读取时的源数据不同，bro生成的点云和pypcd读取的点云数据方向就不同

2. 063315路端没有 从联合标注中剔除
划分：
- 单车端train
- 单路端test testA
- 联合 train 删除？

单路端数据集中没有
联合标注有
split.coop.train有

相当于多创造了一个路端没有的样例

3. 制作前融合数据集kitti create时 014338 样本有错误，从imagesets中剔除
```
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
```

4. 前融合训练集 由所有联合标注组成，根据split里的train val划分为训练验证

5. 训练时用的验证为路端传输的所有点云，评估测试时可以自定义过滤方法，因此二者有一定差异，使用评估时的指标，训练时的验证指标仅作参考

6. 问题： 如果自己训练前融合模型，训练集和 评估的验证集不同，是否会影响效果？
7. 前融合数据集的验证集可视化有问题


## compare socres(Rectangle center, Rectangle filt)
- 使用box中心更准

> 见图，
在低点云数据传输的要求下，给予置信度高的box大的范围效果更好
在高点云数据传输的要求下，给予置信度低的box大的范围效果更好
总体而言，数据传输越多，效果越好?
低数据要求下，我们应该确保传输的数据更加有效，因此更倾向传输置信度高的box
高数据要求下，传输数据够多，可以保证置信度高的box传输足够的有效数据，同时，置信度较高的box预测较准，在较小的范围内即可覆盖有效数据，而置信度较低的box可能与真实box有偏差，因此基于较大的范围有利于提升精度。

上述方法失败，可根据肘部法选取最佳点，并标注出base前融合的点作为比较
### positive
> range = size/2 * scores * k

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|
|-|-|-|-|-|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|62.50|1|12946.80|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|65.31|2|42317.47|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.87|3|73088.27|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.38|4|112148.67|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.84|5|154918.13|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.33|6|193681.20|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.29|7|241272.67|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.52|8|299966.27|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.64|9|357191.60|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.73|10|412541.47|

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|
|-|-|-|-|-|-|-|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|51.86|1|8483.73|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|52.96|2|38387.07|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.28|3|73392.00|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.28|4|112760.40|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.47|5|154136.27|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.73|6|192660.93|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.81|7|234559.47|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.64|8|286779.07|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.38|9| 344278.27|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|54.36|10|403690.40|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|53.73|11|449994.00|


|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|
|-|-|-|-|-|-|-|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|62.01|1|8483.7|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|64.15|2|38387.07|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|65.69|3|73392.00|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.46|4|112760.407|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.81|5|154136.27|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.79|6|192660.93|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.01|7|234559.47|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.13|8|286779.07|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.56|9|344278.27|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.49|10|403690.40|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.75|11|449994.00|


### negative
> range = size/2 * (1/scores) * k

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|
|-|-|-|-|-|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|59.55|0.5|10599.20|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|64.04|0.8|36633.20 |
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|64.22|1|59758.40|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.02|2|174722.80|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.99|2.5|245263.47|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|69.19|3|321401.20|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|69.33|4|443917.87|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.84|5|154918.13|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.33|6|193681.20|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.29|7|241272.67|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.52|8|299966.27|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.64|9|357191.60|


|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|
|-|-|-|-|-|-|-|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|49.58|0.3|4639.07|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|49.72|0.4|8773.73|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|50.25|0.5|14449.60|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|52.06|0.5|42100.40|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|52.59|1|66750.80|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|53.22|1.5|125854.93|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|54.66|2|188686.67|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.38|2.5|256824.53|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|54.99|3|323556.80|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|54.82|3.5|380658.00|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|54.33|4|435727.73|

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|
|-|-|-|-|-|-|-|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|59.83|0.5|14449.60|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|62.21|0.8|42100.40 |
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|62.61|1|66750.80|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|65.10|1.5|125854.93|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.13|2|188686.67|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.68|2.5|256824.53|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.88|3|323556.80|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.68|4|435727.73|


# Report Table
## compare models
### inf
| view |model|eval_dataset | mAP|
|-|-|-|-|
|inf|pointpillars_inf_3_vic_inf_base| vic_inf_val| 54.4083|
|inf|pointpillars_inf_3_vic_inf| vic_inf_val| 62.7306|
|inf|pointpillars_inf_1_vic_inf| vic_inf_val| 62.3176|
|-|-|-|-|
|inf|second_inf_3_sv_inf_base| vic_inf_val|36.1881|
|inf|second_inf_1_vic_inf| vic_inf_val|44.6465 |
|-|-|-|-|
|inf|3dssd_inf_1_vic_inf| vic_inf_val|43.3176|
|-|-|-|-|
|inf|mvxnet_inf_3_sv_inf_base| vic_inf_val|35.6908|
|inf|mvxnet_inf_3_vic_inf| vic_inf_val|44.5898|

### pointpillars
|view|inf_model|veh_model|eval_dataset|mAp|ab_cost|
|-|-|-|-|-|-|
|veh_only|-|pointpillars_veh_1_vic_coop|vic_coop_val_15|59.04|0|
|ef_all|-|pointpillars_veh_1_vic_coop|vic_coop_val_15|69.56|955363.60|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.49|57730.80|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.91|58359.20|
### second
|view|inf_model|veh_model|eval_dataset|mAp|ab_cost|
|-|-|-|-|-|-|
|veh_only|-|second_veh_1_vic_coop|vic_coop_val_15|49.53|0|
|ef_base|-|second_veh_1_vic_coop|vic_coop_val_15|51.1|955363.60|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|57.23||
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.72||
### 3dssd
|view|inf_model|veh_model|eval_dataset|mAp|ab_cost|
|-|-|-|-|-|-|
|veh_only|-|3dssd_veh_1_vic_coop|vic_coop_val_15|47.05|0|
|ef_base|-|3dssd_veh_1_vic_coop|vic_coop_val_15|49.38|955363.60|
|ef|pointpillars_inf_1_vic_inf|3dssd_veh_1_vic_coop| vic_coop_val_15|54.08||
|ef|second_inf_1_vic_inf|3dssd_veh_1_vic_coop| vic_coop_val_15|52.70|58359.20|
### mvxnet
|view|inf_model|veh_model|eval_dataset|mAp|ab_cost|
|-|-|-|-|-|-|
|veh_only|-|pointpillars_veh_1_vic_coop|vic_coop_val_15|50.64|0|
|ef_base|-|pointpillars_veh_1_vic_coop|vic_coop_val_15|56.21|955363.60|
|ef|pointpillars_inf_1_vic_inf|mvxnet_veh_3_vic_coop| vic_coop_val_15|56.22||
|ef|second_inf_1_vic_inf|mvxnet_veh_3_vic_coop| vic_coop_val_15|55.63||



> range = (size/2) / weights * k

|view|inf_model|veh_model|eval_dataset|mAP|k|ab_cost|0.3|0.7|
|-|-|-|-|-|-|-|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|59.63|0.5|5325.07|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|61.27|0.6|8637.87|65.98|52.35|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|64.00|0.7|13529.07|68.30|54.83|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|65.88|0.8|21490.13 |-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.81|0.9|33491.73 |-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15| 66.98|1|40050.53|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.34|1.5|68984.00|70.80|59.03|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.64|2|97590.27|71.08|59.38|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.20|2.5|130454.00|71.27|59.77|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.75|3|165810.80|71.59|59.80|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.50|3.5|201105.33|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.70|4.5| 287026.13|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.82|5|330502.53|71.80|59.99|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|69.43|6|420860.00|72.04|60.04|


# sp(支持)
> range = size/2 * k

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|
|-|-|-|-|-|-|-|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|59.81|0.6|4079.33|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|64.49|0.8|11504.80|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|65.41|1|28335.87|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.90|1.5|46583.33|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.90|2|61524.53|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.93|3|95363.47|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.23|4|141641.73|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.16|5|186484.27|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.03|6|236614.80|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.32|8|379190.53|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.80|10|510921.60|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.35|12|603691.20|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.98|16|734096.80|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|69.56|-|955363.60|
> range = size/2 * weights * k

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|0.3|0.7|
|-|-|-|-|-|-|-|-|-|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|61.17|0.7|4404.27|-|-|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|62.97|0.8|7572.80|-|-|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|64.53|0.9|13890.40|-|-|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|64.06|1|21310.67|-|-|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|65.04|1.2|30232.27|68.61|56.26|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.39|1.5|39565.47|69.67|57.23|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.59|2|53644.807|69.95|57.25|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.90|3|81435.20|70.30|57.76|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.76|4|114642.53|-|-|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.15|5|154126.27|70.58|58.07|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.23|6|194729.60|70.70|28.08|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|-|7|234559.47|-|-|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.61|8|304273.87|-|-|


> range = (size/2)*(1.5-weights) * k

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|0.3|0.7|
|-|-|-|-|-|-|-|-|-|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|60.89|1|7147.60|66.10|52.38|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|63.05|1.2|10792.00|67.31|53.66|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|64.94|1.5|18960.00|68.80|55.32|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.62|2|37972.53|69.87|56.96|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.93|2.5|50689.20|69.88|57.17|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.06|3|62111.07|70.04|57.46|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.27|4|87707.20|70.58|58.06|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.06|5|115503.07|70.59|57.92|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.07|6|146000.27|70.39|58.10|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.14|7|177673.20|70368|57.88|



> range = size/2 * weights * k

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|0.3|0.7|
|-|-|-|-|-|-|-|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|62.44|0.7|5898.13|66.72|53.28|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|63.96|0.8|10229.33|67.93|54.75|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|65.45|0.9|18760.27|68.61|56.49|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.35|1|26970.00|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.00|1.5|47342.93|70.80|58.57|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.21|2|63246.27|70.82|59.20|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.07|3|96702.40|71.10|59.57|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.35|4|142158.67|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.51|5|186383.07|71.47|59.85|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.58|6|234958.80|71.47|60.05|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.69|7|297215.73|71.46|60.07|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.70|8|364291.47|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.73|9|432564.80|71.37|60.10|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|-|10|412541.47|-|-|



> range = (size/2)*(1.5- weights) * k

|view|inf_model|veh_model|eval_dataset|mAP|k|ab_cost|0.3|0.7|
|-|-|-|-|-|-|-|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|59.66|1|6078.67|64.70|52.08|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|65.67|1.5|19437.33|69.43|57.48|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.94|2|42054.67|70.18|58.02|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.05|2.5|57837.60|70.62|59.09|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.32|3|71918.80|70.89|58.99|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.71|3.5|86409.20|71.16|59.74|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.71|4|101864.00|71.22|59.70|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.44|4.5| 119074.67|71.19|59.73|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.46|5|137300.40|71.44|59.68|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.62|6| 173956.67|71.52|59.86|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.48|6| 211291.47|71.57|59.91|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.82|9| 304269.87|71.87|59.98|

# sm(支持)
> range = size/2 * k

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|
|-|-|-|-|-|-|-|
|ef|second_inf_1_vic_inf|mvxnet_veh_3_vic_coop| vic_coop_val_15|51.65|0.6|4079.33|
|ef|second_inf_1_vic_inf|mvxnet_veh_3_vic_coop| vic_coop_val_15|54.13|0.8|11504.80|
|ef|second_inf_1_vic_inf|mvxnet_veh_3_vic_coop| vic_coop_val_15|55.08|1|28335.87|
|ef|second_inf_1_vic_inf|mvxnet_veh_3_vic_coop| vic_coop_val_15|55.20|2|61524.53|
|ef|second_inf_1_vic_inf|mvxnet_veh_3_vic_coop| vic_coop_val_15|55.49|3|95363.47|
|ef|second_inf_1_vic_inf|mvxnet_veh_3_vic_coop| vic_coop_val_15|55.50|5|186484.27|
|ef|second_inf_1_vic_inf|mvxnet_veh_3_vic_coop| vic_coop_val_15|55.88|8|379190.53|
|ef|second_inf_1_vic_inf|mvxnet_veh_3_vic_coop| vic_coop_val_15|56.18|12|603691.33|
|ef|second_inf_1_vic_inf|mvxnet_veh_3_vic_coop| vic_coop_val_15|56.21|-|955363.60|
