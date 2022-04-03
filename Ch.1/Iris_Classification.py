# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("iris_dataset의 키:\n", iris_dataset.keys())

print(iris_dataset['DESCR'][:193] + "\n...")

print("타깃의 이름:", iris_dataset['target_names'])

print("특성의 이름?\n", iris_dataset['feature_names'])

print("data의 타입:", type(iris_dataset['data']))

print("data의 크기:", iris_dataset['data'].shape)

print("data의 처음 다섯 행:\n", iris_dataset['data'][:5])

print("target의 타입:", type(iris_dataset['target']))

print("target의 크기:", iris_dataset['target'].shape)

print("타깃:\n", iris_dataset['target'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train 크기:", X_train.shape)
print("y_train 크기:", y_train.shape)

print("X_test 크기:", X_test.shape)
print("y_test 크기:", y_test.shape)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)

prediction=knn.predict(X_new)
print("예측:", prediction)
print("예측한 타깃의 이름:", iris_dataset['target_names'][prediction])

y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n", y_pred)

print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))

print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))
