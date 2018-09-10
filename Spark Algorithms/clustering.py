# Reference https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/

import time
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark import SparkConf, SparkContext, SQLContext

conf = SparkConf().setMaster("local[*]")
sc = SparkContext(conf=conf)
spark = SQLContext(sc)

start_time = time.time()

data = spark.read.format("libsvm").load("D:\Outils\Spark\data\mllib\iris_libsvm.txt")

model = KMeans().setK(3)

model = model.fit(data)
predictions = model.transform(data)

evaluator = ClusteringEvaluator()

print("\ntime elapsed: {:.2f}s".format(time.time() - start_time))
score = evaluator.evaluate(predictions)
print('Accuracy: ', score)