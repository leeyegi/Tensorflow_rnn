import math
import numpy as np
import matplotlib.pyplot as plt

# z는 값(scalar)일 수도 있고, vector 또는 matrix일 수도 있다.
def sigmoid(z):
    return 1 / (1 + math.e ** -z)

def costFunction(W, X, y):

    m = y.size                  # 100

    # 최초 실행시 값 : [[ 0.5] [ 0.5] [ 0.5] ... [ 0.5]]
    h = sigmoid(np.dot(W, X))   # 1행 m열

    # 값 1개. 곱셈(*)은 element-wise 곱셈
    cost = -(1/m) * sum(y*np.log(h) + (1-y)*np.log(1-h))

    # (h-y)는 1행 m열
    grad = (1/m) * np.dot(X, h-y)

    return cost, grad

# ex2data1.txt 파일에는 아래와 같은 줄이 100개 있다. 쉼표로 구분할 수 있는 csv 파일.
# 34.62365962451697,78.0246928153624,0
# 30.28671076822607,43.89499752400101,0
# 35.84740876993872,72.90219802708364,0
# 60.18259938620976,86.30855209546826,1
# 79.0327360507101,75.3443764369103,1

xy = np.loadtxt('ex2data1.txt', unpack=True, dtype='float32', delimiter=',')

print(xy.shape)     # (3, 100). 행과 열을 바꿔서 읽어온다.
print(xy[:,:5])     # numpy 문법. 리스트는 안됨

# [[ 34.62366104  30.28671074  35.84740829  60.18259811  79.03273773]
#  [ 78.02469635  43.89499664  72.90219879  86.3085556   75.34437561]
#  [  0.           0.           0.           1.           1.        ]]

x_data = xy[:-1]                    # 2행 100열. 정확하게는 2차원 배열
y_data = xy[-1]                     # 1행 100열. 정확하게는 1차원 배열

# y_data가 1 또는 0인 값의 인덱스 배열 생성
pos = np.where(y_data==1)
neg = np.where(y_data==0)

# 옥타브와 비슷한 형태로 그래프 출력
# x_data[0,pos]에서 0은 행, pos는 열을 가리킨다. 쉼표 양쪽에 범위 또는 인덱스 배열 지정 가능.
t1 = plt.plot(x_data[0,pos], x_data[1,pos], color='black', marker='+', markersize=7)
t2 = plt.plot(x_data[0,neg], x_data[1,neg], markerfacecolor='yellow', marker='o', markersize=7)

plt.xlabel('exam 1 score')
plt.ylabel('exam 2 score')
plt.legend([t1[0], t2[0]], ['Admitted', 'Not admitted'])        # 범례

plt.show()

# ---------------------------------------------------------------------- #

n, m = x_data.shape         # [2, 100]. 행과 열의 크기
print('m, n :', m, n)

# 1로 구성된 배열을 맨 앞에 추가
x_data = np.vstack((np.ones(m), x_data))
print(x_data.shape)         # 100
print(x_data[:,:5])

# [[  1.           1.           1.           1.           1.        ]
#  [ 34.62366104  30.28671074  35.84740829  60.18259811  79.03273773]
#  [ 78.02469635  43.89499664  72.90219879  86.3085556   75.34437561]]

W = np.zeros(n+1)           # [ 0.  0.  0.]. 1행 3열

cost, grad = costFunction(W, x_data, y_data)
print('------------------------------')
print('cost :',  cost)      # cost : 0.69314718056
print('grad :', *grad)      # grad : -0.1 -12.0092164707 -11.2628421021


