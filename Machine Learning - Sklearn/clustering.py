# Reference https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/

from sklearn.datasets import load_iris
from sklearn import metrics
import pandas as pd
import numpy as np
import time

from sklearn.cluster import KMeans


data = load_iris()
dataframe = pd.DataFrame(data.data, columns=data.feature_names)
dataframe['species'] = pd.Categorical.from_codes(data.target, data.target_names)
dataframe['is_train'] = np.random.uniform(0, 1, len(dataframe)) <= .75

train, test = dataframe[dataframe['is_train']==True], dataframe[dataframe['is_train']==False]
features = dataframe.columns[:4]

y = pd.factorize(train['species'])[0]

model = KMeans(n_clusters=3)

model.fit(train[features])
predictions = model.labels_

score = metrics.adjusted_rand_score(y, predictions)
print('Accuracy: ', score)