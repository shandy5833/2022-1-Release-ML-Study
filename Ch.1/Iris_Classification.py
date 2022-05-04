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

# ## 1.7 첫 번째 애플리케이션: 붓꽃의 품종 분류
# - 3가지 종으로 분류한 붓꽃의 측정 데이터가 주어짐  
#   이 측정값을 이용해서 임의의 붓꽃이 어떤 품종인지 구분  
#
#   
# - 붓꽃의 품종을 정확하게 분류한 데이터를 가지고 있으므로 __지도 학습__에 속함  
#   3가지 종 중 하나를 선택하는 문제이므로 __분류__ 문제에 해당  
#   이 때 출력될 수 있는 값(붓꽃의 종류)들을 __클래스__라고 함
#
#   
# - 데이터 포인트 하나에 대한 기대 출력 : 꽃의 품종  
#   특정 데이터 포인트에 대한 출력, 즉 품종을 __레이블__이라고 함  

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# ### 1.7.1 데이터 적재
# - 머신러닝과 통계 분야에서 오래전부터 사용해온 붓꽃 데이터셋

from sklearn.datasets import load_iris
iris_dataset = load_iris()

# - iris 객체는 파이썬의 딕셔너리와 유사한 Bunch 클래스의 객체 : 키와 값으로 구성

print("iris_dataset의 키:\n", iris_dataset.keys())

# - DESCR 키에는 데이터셋에 대한 간략한 설명이 들어 있음

print(iris_dataset['DESCR'][:193] + "\n...")

# - target_names : 예측하려는 붓꽃 품종의 이름의 문자열 배열
#
#   
# - feature_names : 각 특성을 설명하는 문자열 리스트

print("타깃의 이름:", iris_dataset['target_names'])
print("특성의 이름:\n", iris_dataset['feature_names'])

# - 실제 데이터는 target과 data 필드에 들어 있음
# - data : 꽃잎의 길이와 폭, 꽃받침의 길이와 폭을 수치 값으로 가지고 있는 NumPy 배열

print("data의 타입:", type(iris_dataset['data']))

# - 머신러닝에서 각 아이템은 __샘플__이라 하고, 속성은 __특성__이라 부름  
#   -> data 배열의 크기는 샘플의 수에 특성의 수를 곱한 값
#
#   
# - scikit-learn은 항상 데이터가 이런 구조일 것이라 가정

print("data의 크기:", iris_dataset['data'].shape)

print("data의 처음 다섯 행:\n", iris_dataset['data'][:5])

# - target : 샘플 붓꽃의 품종을 담은 NumPy 배열

print("target의 타입:", type(iris_dataset['target']))

print("target의 크기:", iris_dataset['target'].shape)

print("타깃:\n", iris_dataset['target'])

# ### 1.7.2 성과 측정: 훈련 데이터와 텍스트 데이터
# - 모델을 데이터에 적용하기 전에, 모델의 예측을 신뢰할 수 있는지 알아야 함  
#
#   
# - 모델을 만들 때 쓴 데이터는 훈련 목적으로 사용할 수 없음  
#   - 모델이 훈련 데이터를 전부 기억할 수 있기 때문  
#
#   
# - 모델의 성능을 측정하려면 학습할 때 본 적 없는 데이터를 모델에 적용해야 함  
#   -> 레이블된 데이터를 두 그룹으로 분리  
#   - __훈련 데이터__ 혹은 __훈련 세트__ : 머신러닝 모델을 만들 때 사용  
#   - __테스트 데이터__, __테스트 세트__, 혹은 __홀드아웃 세트__ : 모델이 얼마나 잘 작동하는지 측정하는 데 사용  
#
#   
# - train_test_split : 데이터셋을 섞어서 나눠 줌  
# - 전체 데이터 중 75%를 훈련 세트로, 25%를 테스트 세트로 뽑음  
# - 데이터를 X로, 레이블을 y로 표기  
# - 유사 난수 생성기에 넣을 난수 초깃값을 random_state 매개변수로 전달  
#   -> 코드가 항상 같은 결과를 출력

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train 크기:", X_train.shape)
print("y_train 크기:", y_train.shape)

print("X_test 크기:", X_test.shape)
print("y_test 크기:", y_test.shape)

# ### 1.7.3 가장 먼저 할 일: 데이터 살펴보기
# - 필요한 정보가 누락되지는 않았는지, 데이터에서 튀는 값이 있는지 찾아야 함  
#
#   
# - 산점도 : 데이터를 시각화  
#   - 산점도는 한 번에 3개 이상의 특성을 표현하기 어려움  
#     -> 모든 특성을 짝지어 만드는 산점도 행렬을 사용

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 iris_dataset.feature_names에 있는 문자열을 사용합니다.
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

# ### 1.7.4 첫 번째 머신러닝 모델: k-최근접 이웃 알고리즘
# - 데이터 포인트가 주어지면, 그 데이터 포인트에서 가장 가까운 훈련 데이터 포인트를 탐색  
#
#   
# - k-최근접 이웃 알고리즘 : 데이터 포인트에서 가장 가까운 'k개'의 이웃을 찾음
#   -> 빈도가 가장 높은 클래스를 예측값으로 사용  
#
#   
# - n_neighbors : 이웃의 개수를 나타내는 매개변수

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# - 훈련 데이터셋으로부터 모델을 만들기 위해 knn 객체의 fit 메서드를 사용

knn.fit(X_train, y_train)

# ### 1.7.5 예측하기
# - 만들어진 모델을 사용해서 정확한 레이블을 모르는 새 데이터에 대해 예측을 만들 수 있음  
#   
#   
# - 새로운 테스트 데이터를 NumPy 배열로 생성

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)

# - scikit-learn은 항상 데이터가 2차원 배열일 것으로 예상

prediction=knn.predict(X_new)
print("예측:", prediction)
print("예측한 타깃의 이름:", iris_dataset['target_names'][prediction])

# ### 1.7.6 모델 평가하기
# - 앞서 만든 테스트 세트를 사용
# - 테스트 세트에 있는 각 붓꽃의 품종을 정확히 알고 있음
# - 얼마나 많은 붓꽃 품종이 정확히 맞았는지 정확도를 계산하여 모델의 성능을 평가

y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n", y_pred)

print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))

print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))

# ## 1.8 요약 및 정리
# - 지도 학습과 비지도 학습의 차이
#   
#   
# 붓꽃의 품종이 무엇인지 예측하는 작업
# - 정확한 품종으로 구분해 놓은 데이터셋을 사용했으므로 __지도 학습__에 해당  
#   
#   
# - 품종이 세 개이므로 세 개의 클래스를 분류하는 문제  
#   
#   
# - 각 품종을 __클래스__, 개별 붓꽃의 품종은 __레이블__이라고 함  
#   
#   
# - 붓꽃 데이터셋은 두 개의 NumPy 배열 - X와 y - 로 이루어져 있음
#   - X : 데이터. 특성들의 2차원 배열이므로 각 데이터 포인트는 행 하나로 나타남
#   - y : 기대하는 출력. 1차원 배열로 각 샘플의 클래스 레이블에 해당하는 정수를 담고 있음
#   
#   
# - 데이터셋을 모델 구축에 사용할 __훈련 세트__와 모델이 새로운 데이터에 얼마나 잘 적용될 수 있을지 평가하기 위한 __테스트 세트__로 나눔
#   
#   
# - __k-최근접 이웃 분류__ 알고리즘은 새 데이터 포인트를 예측하기 위해 훈련 데이터에서 가장 가까운 이웃을 선택
#   - n_neighbors 매개변수를 지정해 이 클래스의 객체를 생성
#   - 훈련 데이터(X_train)와 훈련 데이터의 레이블(y_train)을 매개변수로 하여 fit 메서드를 호출해 모델 생성
#   - 모델의 정확도를 계산하는 score 메서드로 모델을 평가
