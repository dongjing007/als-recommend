## 📚 依赖
* python > 3.5+
* pip install pandas
* pip install pyspark

## 🤔 介绍

![image](https://github.com/dongjing007/als-recommend/assets/21094836/5a708948-4a63-4143-9107-8801b786eb7a)

ALS（alternating least squares）：基于模型的推荐算法，基本思想是对稀疏矩阵进行模型分解，评估出缺失项的值，以此来得到一个基本的训练模型。

ALS是采用交替的最小二乘法来算出缺失项的，交替的最小二乘法是在最小二乘法的基础上发展而来的。

ALS目前集成到Spark里面，使用非常便利。

## 💁 应用
[十分钟入门机器学习 - 讲透ALS推荐算法](https://juejin.cn/post/7002793334573891615)

程序通过已有的 users，items，score 数据矩阵生成模型，预测新用户对物品的喜好度。



