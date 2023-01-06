# Importing the libraries
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
iris = datasets.load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Extracting the features and target
x = df.drop('target', axis=1)
y = df['target']

# Defining the distance function


def distance(a, b, p=2):

    dim = len(a)
    distance = 0
    for d in range(dim):
        distance += abs(a[d] - b[d])**p
    distance = distance**(1/p)
    return distance


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Feature Scaling
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# Defining the KNN classifier


def knn_predict(X_train, X_test, y_train, y_test, k, p):
    test_preds = []
    for test_point in X_test:
        distances = []
        for train_point in X_train:
            distance_ = distance(test_point, train_point, p=p)
            distances.append(distance_)
        df_dists = pd.DataFrame(data=distances, columns=['dist'],
                                index=y_train.index)
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]
        counter = Counter(y_train[df_nn.index])
        prediction = counter.most_common()[0][0]
        test_preds.append(prediction)

    return test_preds


# Predicting the Test set results
test_preds = knn_predict(X_train, X_test, y_train, y_test, k=5, p=2)

# Printing the accuracy
print(f"Accuracy: {sum(test_preds == y_test)/len(y_test)*100:.2f}%")
