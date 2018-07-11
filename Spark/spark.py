# Reference https://github.com/cerndb/dist-keras/blob/master/examples/mnist.py

import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.databricks:spark-csv_2.10:1.4.0 pyspark-shell'
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark import SQLContext

application_name = "Spark_App"
master = "local[*]"
num_processes = 2
num_executors = 1
num_workers = num_executors * num_processes

conf = SparkConf()
conf.set("spark.app.name", application_name)
conf.set("spark.master", master)
conf.set("spark.executor.cores", num_processes)
conf.set("spark.executor.instances", num_executors)
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")
conf.set("spark.locality.wait", "0")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
reader = sqlContext

path_train = "mnist_train.csv"
path_test = "mnist_test.csv"

raw_dataset_train = reader.read.format('com.databricks.spark.csv') \
    .options(header='true', inferSchema='true') \
    .load(path_train)
raw_dataset_test = reader.read.format('com.databricks.spark.csv') \
    .options(header='true', inferSchema='true') \
    .load(path_test)

from pyspark.ml.feature import VectorAssembler
from distkeras.transformers import *

features = raw_dataset_train.columns
features.remove('label')

vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
dataset_train = vector_assembler.transform(raw_dataset_train)
dataset_test = vector_assembler.transform(raw_dataset_test)

encoder = OneHotTransformer(10, input_col="label", output_col="label_encoded")
dataset_train = encoder.transform(dataset_train)
dataset_test = encoder.transform(dataset_test)

transformer = MinMaxTransformer(n_min=0.0, n_max=1.0, o_min=0.0, o_max=250.0, input_col="features",
                                output_col="features_normalized")
dataset_train = transformer.transform(dataset_train)
dataset_test = transformer.transform(dataset_test)

reshape_transformer = ReshapeTransformer("features_normalized", "matrix", (28, 28, 1))
dataset_train = reshape_transformer.transform(dataset_train)
dataset_test = reshape_transformer.transform(dataset_test)

dataset_train = dataset_train.select("features_normalized", "matrix", "label", "label_encoded")
dataset_test = dataset_test.select("features_normalized", "matrix", "label", "label_encoded")

dense_transformer = DenseTransformer(input_col="features_normalized", output_col="features_normalized_dense")
dataset_train = dense_transformer.transform(dataset_train)
dataset_test = dense_transformer.transform(dataset_test)

dataset_train.repartition(num_workers)
dataset_test.repartition(num_workers)

training_set = dataset_train.repartition(num_workers)
test_set = dataset_test.repartition(num_workers)

training_set.cache()
test_set.cache()

if os.path.isfile('spark.json') and os.path.isfile('spark.h5'):
    from keras.models import model_from_json

    json_file = open('spark.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    trained_model = model_from_json(loaded_model_json)
    trained_model.load_weights("spark.h5")
    trained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.layers.core import Flatten, Dense
    from keras.models import Sequential

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    from distkeras.trainers import *

    trainer = ADAG(keras_model=model, worker_optimizer='adam', loss='categorical_crossentropy',
                   num_workers=num_workers, batch_size=16, communication_window=5, num_epoch=5,
                   features_col="matrix", label_col="label_encoded")
    trained_model = trainer.train(training_set)

    print("time elapsed: " + str(trainer.get_training_time()))

    model_json = trained_model.to_json()
    with open("spark.json", "w") as json_file:
        json_file.write(model_json)
    trained_model.save_weights("spark.h5")

from distkeras.evaluators import *
from distkeras.predictors import *

evaluator = AccuracyEvaluator(prediction_col="prediction_index", label_col="label")
predictor = ModelPredictor(keras_model=trained_model, features_col="matrix")
transformer = LabelIndexTransformer(output_dim=10)
test_set = test_set.select("matrix", "label")
test_set = predictor.predict(test_set)
test_set = transformer.transform(test_set)
score = evaluator.evaluate(test_set)

print('Accuracy: ', str(score))
