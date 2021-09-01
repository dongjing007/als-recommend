import pandas as pd
from pyspark.sql.functions import col, explode
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.recommendation import ALSModel,ALS

# 创建spark
sc = SparkContext
spark = SparkSession.builder.appName('Recommendations').getOrCreate()

# 读取csv数据源
scores = spark.read.csv("ratings_match.csv", header=True)
scores = scores.\
    withColumn('userId', col('userId').cast('integer')).\
    withColumn('postId', col('postId').cast('integer')).\
    withColumn('score', col('score').cast('float'))

# 计算数据稀疏度
user_nums = scores.select("userId").distinct().count()
post_nums = scores.select("postId").distinct().count()
known_data_nums = scores.select("score").count()
matrix_item = user_nums * post_nums
sparsity = (1 - (known_data_nums / matrix_item)) * 100
print("  sparsity:", sparsity)

# 创建数据集和测试集
(train, test) = scores.randomSplit([0.8, 0.2], seed=1234)

# 创建ALS模型
als = ALS(userCol="userId",
          itemCol="postId",
          ratingCol="score",
          nonnegative=True,
          implicitPrefs=True,
          coldStartStrategy="drop")

# 设置参数
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [5,10,15,20]) \
            .addGrid(als.regParam, [.01, .05, .1, .15]) \
            .addGrid(als.maxIter, [5, 50, 100, 200]) \
            .build()

# RMSE来设定模型
evaluator = RegressionEvaluator(metricName="rmse",
                                labelCol="score",
                                predictionCol="prediction")

# 需要训练次数
print("Num models to be tested: ", len(param_grid))

# CrossValidator交叉验证
cv = CrossValidator(estimator=als,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    numFolds=5)

model = cv.fit(train)

# 获取到最新模型
best_model = model.bestModel

# 获取最佳模型
print("**Best Model**")
# Print "Rank"
print("  Rank:", best_model._java_obj.parent().getRank())
# Print "MaxIter"
print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
# Print "RegParam"
print("  RegParam:", best_model._java_obj.parent().getRegParam())

# 查看预测值
test_predictions = best_model.transform(test)
RMSE = evaluator.evaluate(test_predictions)
print(RMSE)

# 保存最优模型
best_model.save("als_recommend_model")

# 加载模型
ALSModel = ALSModel.load("als_recommend_model")
nrecommendations = ALSModel.recommendForAllUsers(25)
nrecommendations = nrecommendations\
    .withColumn("rec_exp", explode("recommendations"))\
    .select('userId', col("rec_exp.postId"), col("rec_exp.rating"))

# user预测评分
# nrecommendations.filter('userId = xxx').limit(10).show()

# user实际评分
# scores.filter('userId = xxx').sort('score',ascending=False).limit(10).show()

# 导出所有预测值
nrecommendations.coalesce(1).write.csv('mycsv.csv')
