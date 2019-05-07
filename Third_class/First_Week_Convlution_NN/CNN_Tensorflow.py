import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops

import cnn_utils

np.random.seed(1)

def create_placeholders(n_H0, n_W0, n_C0, n_y):

    X = tf.placeholder(tf.float32,[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32,[None, n_y])

    return X,Y

def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1, "W2": W2}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']

    X = tf.nn.conv2d(X, W1, [1, 1, 1, 1], padding="SAME")
    X = tf.nn.relu(X)
    X = tf.nn.max_pool(X, [1, 8, 8, 1], [1, 8, 8, 1], padding="SAME")

    X = tf.nn.conv2d(X, W2, [1, 1, 1, 1], padding="SAME")
    X = tf.nn.relu(X)
    X = tf.nn.max_pool(X, [1, 4, 4, 1], [1, 4, 4, 1], padding="SAME")

    X = tf.layers.flatten(X)
    X = tf.contrib.layers.fully_connected(X, 6, activation_fn=None)

    return X

def compute_cost(Z3,Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.005,
         num_epochs=100,minibatch_size=64,print_cost=True,isPlot=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    X,Y = create_placeholders( n_H0, n_W0, n_C0,n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)

    cost = compute_cost(Z3,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)  # 获取数据块的数量
            seed = seed + 1
            minibatches = cnn_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
            if print_cost:
                # 每5代打印一次
                if epoch % 5 == 0:
                    print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost))
                    # 记录成本
            if epoch % 1 == 0:
                costs.append(minibatch_cost)

                # 数据处理完毕，绘制成本曲线
        if isPlot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        predict_op = tf.arg_max(Z3, 1)
        corrent_prediction = tf.equal(predict_op, tf.arg_max(Y, 1))

        ##计算准确度
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))
        print("corrent_prediction accuracy= " + str(accuracy))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuary = accuracy.eval({X: X_test, Y: Y_test})

        print("训练集准确度：" + str(train_accuracy))
        print("测试集准确度：" + str(test_accuary))

        return (train_accuracy, test_accuary, parameters)


if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cnn_utils.load_dataset()
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
    Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))


    _,_, parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=150)

