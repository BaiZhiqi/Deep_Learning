import numpy as np
import tensorflow as tf
cofficient = np.array([[1.],[-20.],[100.]])
w = tf.Variable(0.,dtype=tf.float32)
x = tf.placeholder(tf.float32,[3,1])
#cost = tf.add(tf.add(w**2,tf.multiply(-10.,w)),25.)
# cost = w**2 + -10.*w +25.
cost = x[0][0]*w**2 + x[1][0]*w +x[2][0]*25.
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as Session:
    Session.run(init)
    print(Session.run(w))
Session.run(train,feed_dict={x:cofficient})
print(Session.run(w))
for i in range(1000):
    Session.run(train,feed_dict={x:cofficient})
print(Session.run(w))