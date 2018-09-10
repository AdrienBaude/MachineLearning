# Reference https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/

import time
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, MultilayerPerceptronClassifier, LinearSVC, OneVsRest, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkConf, SparkContext, SQLContext

conf = SparkConf().setMaster("local[*]")
sc = SparkContext(conf=conf)
spark = SQLContext(sc)

start_time = time.time()

data = spark.read.format("libsvm").load("D:\Outils\Spark\data\mllib\iris_libsvm.txt")
(train, test) = data.randomSplit([0.8, 0.2])

model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = MultilayerPerceptronClassifier(layers=[4, 3])
# model = OneVsRest(classifier=LinearSVC())
# model = NaiveBayes()

model = model.fit(train)
predictions = model.transform(test)

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

print("\ntime elapsed: {:.2f}s".format(time.time() - start_time))
score = evaluator.evaluate(predictions)
print('Accuracy: ', score)
