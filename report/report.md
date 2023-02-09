
## 优化三部曲
### 1. 选定路端模型，确定用哪几个模型，每个模型使用vic_inf_train训练，以 **vic_inf_val**为评估数据集
模型：
- **pointpillars**
- **second**
- 3dssd
- mvxnet
### 2. 选定车端模型，确定用那哪几个模型，使用coop前融合_train，以**coop**_过滤融合_val为评估数据集(以路端最优模型作为过滤 )
- **pointpillars**
- **second**
- mvxnet
- 3dssd

### 3.车路端模型组合，得到组合评估表
### 4.优化融合策略，加入score作为范围划定标准
- 加入score效果下降
### **5.优化前融合训练集，使用过滤后的inf点云融合**，过滤方法见转化脚本注释
**- 先搞一个训练集，提升一下基础精度，然后做出cost-mAP图像肘部法选点**
### 6.再搞一个高精度的路端模型

# IDEA
- 路端所用到的其实只有小部分有效点，见图



# Report Table
## compare models(Box center,  Rectangle filt)
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

### pointpillars(no weights)
|view|inf_model|veh_model|eval_dataset|mAp_BEV_0.5|ab_cost|mAp_BEV_0.3|mAp_BEV_0.7|
|-|-|-|-|-|-|-|-|
|veh_only|-|pointpillars_veh_1_vic_coop|vic_coop_val_15|59.04|0|63.50|51.69|
|ef_all|-|pointpillars_veh_1_vic_coop|vic_coop_val_15|69.56|955363.60|72.17|60.24|
|ef|**pointpillars_inf_1_vic_inf**|**pointpillars_veh_1_vic_coop**| **vic_coop_val_15**|**68.28**|**72456.13**|70.92|59.16|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.90|46583.33|70.11|56.99|
66.90	2	61524.53	70.10	57.74
### second
|view|inf_model|veh_model|eval_dataset|mAp|ab_cost|mAp_BEV_0.3|mAp_BEV_0.7|
|-|-|-|-|-|-|-|-|
|veh_only|-|second_veh_1_vic_coop|vic_coop_val_15|49.53|0|54.77|43.38|
|ef_base|-|second_veh_1_vic_coop|vic_coop_val_15|51.1|955363.60|56.15|40.76|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|56.60	|53742.13|60.50|48.19|
|ef|**second_inf_1_vic_inf**|**second_veh_1_vic_coop**|**vic_coop_val_15**|**55.50**|**61524.53**|59.47|47.56|	
## compare socres(Rectangle center, Rectangle filt)
- 使用box中心更准

IDEA：
- 一种比较随点云增加map的变化，做出all横线
# pp(支持)
> range = size/2 * k

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|0.3|0.7|
|-|-|-|-|-|-|-|-|-|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|59.04|0|0|63.50|51.69
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|60.51|0.6|4570.27|65.27|52.22|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|64.91|0.8|13567.33|69.11|55.65|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.84|1|32422.00|70.27|58.08|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.03|1.5|53742.13|70.86|58.48|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.28|2|72456.13|70.92|59.16|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.58|3|114633.47|71.49|59.87|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.65|4|170560.80|71.56|60.15|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.59|5| 225051.20|71.56|59.83|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|69.09|8|433990.67|71.81|60.22|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|69.46|12|650351.87|72.13|60.53|
|ef|pointpillars_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|69.56|-|955363.60|72.17|60.24

# sp(支持)
> range = size/2 * k

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|0.3|0.7|
|-|-|-|-|-|-|-|-|-|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|59.04|0|0|63.50|51.69
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|59.81|0.6|4079.33|64.72|51.85|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|64.49|0.8|11504.80|68.59|55.46|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|65.41|1|28335.87|69.20|56.07|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.90|1.5|46583.33|70.11|56.99|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.90|2|61524.53|70.10|57.74|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|66.93|3|95363.47|70.36|57.99|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.23|4|141641.73|70.54|58.10|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.16|5|186484.27|70.54|57.88|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.03|6|236614.80|70.32|57.90|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.32|8|379190.53|70.50|58.40|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|67.80|10|510921.60|70.92|58.72|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|68.35|12|603691.20|71.27|59.19|
|ef|second_inf_1_vic_inf|pointpillars_veh_1_vic_coop| vic_coop_val_15|69.56|-|955363.60|72.17|60.24



# ss(支持)
> range = size/2 * k

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|0.3|0.7|
|-|-|-|-|-|-|-|-|-|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|49.53|0|0|54.77|43.38|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|50.72|0.6|4079.33|56.09|43.52|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|54.00|0.8|11504.80|58.83|46.07|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|54.95|1|28335.87|59.20|46.16|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.50|2|61524.53|59.47|47.56|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.50|3|95363.60|59.47|47.29|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.65|4|141641.87|59.61|47.54|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.67|6|236614.80|59.76|47.57|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.43|8|379190.53|59.67|47.19|
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|53.50|10|510921.60|58.36|44.19
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|52.39|12|603691.20|57.75|42.52
|ef|second_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|51.1|-|955363.60|56.15|40.76|

# ps(支持)
> range = size/2 * k

|view|inf_model|veh_model|eval_dataset|mAp|k|ab_cost|0.3|0.7|
|-|-|-|-|-|-|-|-|-|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|49.53|0|0|54.77|43.38|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|50.94|0.6|4570.27|56.45|43.88|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.01|0.8|13567.33|59.30|46.06|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|56.22|1|32422.00|60.38|47.17|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|56.60|1.5|53742.13|60.50|48.19|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|56.84|2|72456.13|60.69|48.38|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|57.40|3|114633.47|61.05|48.35|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|57.16|4|170560.80|60.96|48.26|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|57.12|5| 225051.20|60.98|48.26|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|55.55|8|433990.67|59.84|46.50|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|52.21|12|650351.87|57.11|41.91|
|ef|pointpillars_inf_1_vic_inf|second_veh_1_vic_coop| vic_coop_val_15|51.1|-|955363.60|56.15|40.76|