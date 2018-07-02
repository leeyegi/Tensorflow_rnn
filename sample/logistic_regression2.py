import tensorflow as tf
import numpy as np

# 04train.txt
# #x0 x1 x2 y
# 1   2   1   0
# 1   3   2   0
# 1   3   5   0
# 1   5   5   1
# 1   7   5   1
# 1   2   5   1

# 원본 파일은 6행 4열이지만, 열 우선이라서 4행 6열로 가져옴
xy = np.loadtxt('04train.txt', unpack=True, dtype='float32')

#입력데이터
x_data = xy[:-1]            # 3행 6열
y_data = xy[-1]             # 1행 6열
print(x_data)
print(y_data)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# feature별 가중치를 난수로 초기화. feature는 bias(편향) 포함해서 3개. 1행 3열.
W= tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# 행렬 곱셈. (1x3) * (3x6)
h = tf.matmul(W, X)

#가설  - H(X)
hypothesis = tf.div(1., 1. + tf.exp(-h))    # exp(-h) = e ** -h. e는 자연상수

# exp()에는 실수만 전달
# print(tf.exp([1., 2., 3.]).eval())      # [2.71828175 7.38905621 20.08553696]
# print(tf.exp([-1., -2., -3.]).eval())   # [0.36787945 0.13533528 0.04978707]

#비용함수 - C(X)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

rate = tf.Variable(0.1)

#경사 하강법 - W
optimizer = tf.train.GradientDescentOptimizer(rate)

#경사하강법으로 구한 기울기 값의 가장 작은 값을 저장
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#2,000번 반복
for step in range(2001):

    sess.run(train, feed_dict={X: x_data, Y: y_data})

    #20번 마다 비용함수 출력
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

print('-----------------------------------------')

# 결과가 0 또는 1로 계산되는 것이 아니라 0과 1 사이의 값으로 나오기 때문에 True/False는 직접 판단
print('[1, 2, 2] :', sess.run(hypothesis, feed_dict={X: [[1], [2], [2]]}) > 0.5)
print('[1, 5, 5] :', sess.run(hypothesis, feed_dict={X: [[1], [5], [5]]}) > 0.5)
print('[1, 4, 2] [1, 0, 10] :', end=' ')
print(sess.run(hypothesis, feed_dict={X: [[1, 1], [4, 0], [2, 10]]}) > 0.5)
sess.close()
