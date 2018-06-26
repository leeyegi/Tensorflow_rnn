import tensorflow as tf
xData = [1,2,3,4,5,6,7]
yData = [25000,55000,75000,110000,128000,155000,180000]

W = tf.Variable(tf.random_uniform([1],-100,100))
b = tf.Variable(tf.random_uniform([1],-100,100))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

H = W * X + b

cost = tf.reduce_mean(tf.square(Y-H))

a = tf.Variable(0.01)

optimizer = tf.train.GradientDescentOptimizer(a)

train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(4001):
    sess.run(train, feed_dict={X:xData,Y:yData})
    if i% 100 == 0:
        print("%4d번째"%i,":","%9.3f"%sess.run(W)[0]," b:" , sess.run(b))

print("8시간의 매출 예측 : ",sess.run(H, feed_dict={X:8}))