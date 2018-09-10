# Reference https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

start_time = time.time()

data = load_iris()
dataframe = pd.DataFrame(data.data, columns=data.feature_names)
dataframe['species'] = pd.Categorical.from_codes(data.target, data.target_names)
dataframe['is_train'] = np.random.uniform(0, 1, len(dataframe)) <= .75

train, test = dataframe[dataframe['is_train']==True], dataframe[dataframe['is_train']==False]
features = dataframe.columns[:4]

y = pd.factorize(train['species'])[0]

model = LogisticRegression()
# model = RandomForestClassifier()
# model = svm.LinearSVC()
# model = GaussianNB()
# model = tree.DecisionTreeClassifier()
# model = MLPClassifier()

model.fit(train[features], y)
predictions = data.target_names[model.predict(test[features])]

print("\ntime elapsed: {:.2f}s".format(time.time() - start_time))
score = accuracy_score(test['species'], predictions)
print('Accuracy: ', score)
