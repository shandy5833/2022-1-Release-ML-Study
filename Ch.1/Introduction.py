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

# np.array : n개의 행, m개의 열을 가진 array를 생성

# +
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n", x)
# -

# 단위 행렬 : 대각선 원소는 1이고 나머지는 0인 행렬  
# np.eye : 크기가 n인 단위 행렬을 생성

# +
from scipy import sparse

eye = np.eye(4)
print("NumPy 배열:\n", eye)
# -

# 희소 행렬 : 행렬의 값이 대부분 0인 행렬  
# 희소 행렬의 표현 방식 : CSR 포맷, COO 포맷  
# sparse.csr_matrix : CSR 포맷의 희소 행렬로 변환

#CSR 포맷
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy의 CSR 행렬:\n", sparse_matrix)

# np.arange : 반열린구간 [start, stop)에서 step의 크기만큼 일정하게 떨어져 있는 숫자들을 array 형태로 반환  
# sparse.coo_matrix :  COO 포맷의 희소 행렬로 변환

#COO 포맷
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO 표현:\n", eye_coo)

# np.linspace : start와 stop 사이를 일정 간격으로 잘라 1차원 배열을 생성
# plt.plot : 한 배열의 값을 다른 배열에 대응해서 선 그래프를 도시

# +
# %matplotlib inline
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker="x")

# +
import pandas as pd

data = {'Name' : ["John", "Anna", "Peter", "Linda"],
       'Location' : ["New York", "Paris", "Berlin", "London"],
       'Age' : [24, 13, 53, 33]
       }

data_pandas = pd.DataFrame(data)
data_pandas
# -

data_pandas[data_pandas.Age > 30]
