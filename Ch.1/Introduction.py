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

# ## 1.1 왜 머신러닝인가?
#
# - 하드코딩된 규칙만으로는 구현할 수 없는 분야가 많음  
#   ex. 얼굴 인식  
#   -> __머신러닝 사용__
#
# ### 1.1.1 머신러닝으로 풀 수 있는 문제
#
# - 입력과 기대되는 출력이 제공되는 __지도 학습__  
#   ex. 스팸 분류
# - 입력은 주어지지만 출력은 제공되지 않는 __비지도 학습__  
#   ex. 블로그 글의 주제 구분
# - 머신러닝에서의 하나의 개체 혹은 데이터셋의 행을 __샘플__ 또는 __데이터 포인트__, 그리고 샘플의 속성 혹은 데이터셋의 열을 __특성__이라고 한다.
#
# ### 1.1.2 문제와 데이터 이해하기
#
# - 어떤 질문에 대한 답을 원하는가? 가지고 있는 데이터가 원하는 답을 줄 수 있는가?
# - 내 질문을 머신러닝의 문제로 가장 잘 기술하는 방법은 무엇인가?
# - 문제를 풀기에 충분한 데이터를 모았는가?
# - 내가 추출한 데이터의 특성은 무엇이며 좋은 예측을 만들어낼 수 있을 것인가?
# - 머신러닝 애플리케이션의 성과를 어떻게 측정할 수 있는가?
# - 머신러닝 솔루션이 다른 연구나 제품과 어떻게 협력할 수 있는가?

# ## 1.2 왜 파이썬인가?
#
# - 데이터 적재, 시각화, 통계, 자연어 처리, 이미지 처리 등에 필요한 라이브러리
# - 터미널이나 주피터 노트북으로 대화형 모드를 사용할 수 있음  
#   \- _대화형 모드가 스크립트 모드에 비해 갖는 장점이 뭘까?_

# ## ~1.3 scikit-learn~
#
# ## 1.4 필수 라이브러리와 도구들
#
# ### 1.4.1 주피터 노트북
# - 프로그램 코드를 브라우저에서 실행해주는 대화식 환경
# - 탐색적 데이터 분석에 적합  
#   - 확증적 데이터 분석 : 가설을 설정한 후, 수집한 데이터로 가설을 평가하고 추정하는 전통적인 분석  
#   가설 설정 -> 데이터 수집 -> 통계 분석 -> 가설 검증
#   - 탐색적 데이터 분석 : Raw data를 가지고 유연하게 데이터를 탐색하고, 데이터의 특징과 구조로부터 얻은 정보를 바탕으로 통계모형을 만드는 분석  
#   데이터 수집 -> 시각화 탐색 -> 패턴 도출 -> 인사이트 발견

# ### 1.4.2 Numpy
# - 고수준 수학 함수와 유사 난수 생성기를 포함  
#
# numpy.array : n개의 행, m개의 열을 가진 array를 생성

# +
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n", x)
# -

# ### 1.4.3 SciPy
# - 고성능 선형 대수, 함수 최적화, 신호 처리, 특수한 수학 함수와 통계 분포 등을 포함  
#
# scipy.sparse : 희소 행렬 기능 제공  
# 희소 행렬 : 0을 많이 포함한 2차원 배열을 저장할 때 사용  
#   
# 단위 행렬 : 대각선 원소는 1이고 나머지는 0인 행렬  
# np.eye : 크기가 n인 단위 행렬을 생성

# +
from scipy import sparse

# 대각선 원소는 1이고 나머지는 0인 2차원 NumPy 배열을 만듭니다.
eye = np.eye(4)
print("NumPy 배열:\n", eye)
# -

# 희소 행렬을 0이 모두 채워진 2차원 배열로부터 만들지 않음(메모리 문제)  
# 희소 행렬의 표현 방식 : CSR 포맷, COO 포맷  
# CSR 포맷과 COO 포맷의 차이 : https://bit.ly/3vVmAik  
# sparse.csr_matrix : CSR 포맷의 희소 행렬로 변환

# NumPy 배열을 CSR 포맷의 SciPy 희박 행렬로 변환합니다.
# 0이 아닌 원소만 저장됩니다.
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy의 CSR 행렬:\n", sparse_matrix)

# numpy.arange : 반열린구간 [start, stop)에서 step의 크기만큼 일정하게 떨어져 있는 숫자들을 array 형태로 반환  
# numpy.ones : 1로 채워진 array를 생성  
# sparse.coo_matrix :  COO 포맷의 희소 행렬로 변환

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO 표현:\n", eye_coo)

# ### 1.4.4 matplotlib
# - 선 그래프, 히스토그램, 산점도 등  
#
# numpy.linspace : start와 stop 사이를 일정 간격으로 잘라 1차원 배열을 생성  
# plt.plot : 한 배열의 값을 다른 배열에 대응해서 선 그래프를 도시

# +
# %matplotlib inline
import matplotlib.pyplot as plt

# -10에서 10까지 100개의 간격으로 나뉘어진 배열을 생성합니다.
x = np.linspace(-10, 10, 100)
# 사인 함수를 이용하여 y 배열을 생성합니다.
y = np.sin(x)
# plot 함수는 한 배열의 값을 다른 배열에 대응해서 선 그래프를 그립니다.
plt.plot(x, y, marker="x")
# -

# ### 1.4.5 pandas
# - 데이터 처리와 분석을 위한 파이썬 라이브러리
# - SQL처럼 테이블에 쿼리나 조인을 수행할 수 있음.

# +
import pandas as pd

# 회원 정보가 들어간 간단한 데이터셋을 생성합니다.
data = {'Name' : ["John", "Anna", "Peter", "Linda"],
       'Location' : ["New York", "Paris", "Berlin", "London"],
       'Age' : [24, 13, 53, 33]
       }

data_pandas = pd.DataFrame(data)
# 주피터 노트북은 Dataframe을 미려하게 출력해줍니다.
data_pandas
# -

# - 테이블에 질의할 수 있음

# Age 열의 값이 30 이상인 모든 행을 선택합니다.
data_pandas[data_pandas.Age > 30]

# ### 1.4.6 mglearn
# - 간단하게 그림을 그리거나 필요한 데이터를 바로 불러들이기 위해 사용  
#
# 모든 코드는 다음의 라이브러리를 임포트한다고 가정

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# ## ~~1.5 파이썬 2 vs. 파이썬 3~~
# ## ~~1.6 이 책에서 사용하는 소프트웨어 버전~~
