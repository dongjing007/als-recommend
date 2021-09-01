# 协同过滤的ALS算法

------

> ALS（alternating least squares）算法是基于模型的推荐算法：其基本思想是对稀疏矩阵进行模型分解，评估出缺失项的值，以此来得到一个基本的训练模型。

------
### ALS算法概要

ALS已经集成到Spark的Mllib库中，使用起来比较方便。

通过已有的《users，items，score》数据矩阵生成模型，预测新用户对物品的喜好度。

ALS是采用交替的最小二乘法来算出缺失项的。交替的最小二乘法是在最小二乘法的基础上发展而来的。

### ALS安装体验
* python版本 > 3.5
* pip install pandas
* pip install pyspark

### ALS代码解读

详见代码注释

### 数据来源
[数据传送门](https://grouplens.org/datasets/movielens/100k/)

