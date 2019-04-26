import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time
np.random.seed(1)
import matplotlib.image as mpimg # mpimg 用于读取图片
def linear_function():
    np.random.seed(1) #指定随机种子
    X = np.random.randn(3,1)
    W = np.random.randn(4,3)
    b = np.random.randn(4,1)
    Z = tf.add(tf.matmul(W,X),b)
    with tf.Session() as sess:
        result = sess.run(Z)
    sess.close()
    return result

def sigmoid(z):
    x = tf.placeholder(tf.float32,name='x')
    sigmoid = tf.sigmoid(x)
    with tf.Session() as sess:
        result = sess.run(sigmoid,feed_dict={x:z})
    sess.close()
    return result

def one_hot_matrix(lables,C):
    '''
    :param lables: 每个样本的类别的tensor
    :param C: 一共包含多少类别
    :return: 标签的one_hot编码
    '''
    one_hot_matrix = tf.one_hot(indices=lables , depth=C , axis=0)
    with tf.Session() as sess:
        result = sess.run(one_hot_matrix)
    sess.close()
    return result
def ones(shape):

    one = tf.ones(shape)
    with tf.Session() as sess:
        result = sess.run(one)
    sess.close()
    return result
def create_placeholders(n_x,n_y):
    '''
    :param n_x:训练集每个样本的特征长度
    :param n_y:分类的类别
    '''
    X = tf.placeholder(tf.float32,[n_x,None],name='X')
    Y = tf.placeholder(tf.float32, [n_y, None]
    ,name = 'Y')
    return X,Y

def initialize_parameters(layer_dim):
    # import tensorflow as tf
    tf.set_random_seed(1)

    W1 = tf.get_variable('W1',[layer_dim[1],layer_dim[0]],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1',[layer_dim[1],1],initializer=tf.zeros_initializer())

    W2 = tf.get_variable('W2',[layer_dim[2],layer_dim[1]],  initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2',[layer_dim[2],1], initializer=tf.zeros_initializer())

    W3 = tf.get_variable('W3',[layer_dim[3],layer_dim[2]],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3',[layer_dim[3],1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters
def forward_propagation(X,parameters):

    Z1 = tf.add(tf.matmul(parameters["W1"],X),parameters["b1"])
    A1 = tf.nn.relu(Z1)

    Z2 = tf.add(tf.matmul(parameters["W2"], A1), parameters["b2"])
    A2 = tf.nn.relu(Z2)

    Z3 = tf.add(tf.matmul(parameters["W3"], A2), parameters["b3"])


    return Z3


def compute_cost(Z3,Y):
    logits = tf.transpose(Z3)  # 转置
    labels = tf.transpose(Y)  # 转置

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

    return loss
def model(X_train,Y_train,X_test,Y_test,
        learning_rate=0.0001,num_epochs=1500,minibatch_size=32,
        print_cost=True,is_plot=True):
    ops.reset_default_graph()  # 能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape  # 获取输入节点数量和样本数
    n_y = Y_train.shape[0]  # 获取输出节点数量
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters([12288, 25, 12, 6])
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_epochs):
            epoch_cost = 0  # 每代的成本
            num_minibatches = int(m / minibatch_size)  # minibatch的总数量
            seed+=1
            minibatches = tf_utils.random_mini_batches(X_train, Y_train, mini_batch_size=minibatch_size, seed=seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _,minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                epoch_cost += minibatch_cost/num_minibatches
            if i%5==0:
                costs.append(epoch_cost)
                if print_cost and i%100==0:
                    print("epoch ={},epoch_cost ={}".format(i,epoch_cost))
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
        parameters = sess.run(parameters)
        print("参数已经保存到session。")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

if __name__ == '__main__':
    #读取数据
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

    #将图像展开
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    #归一化
    X_train = X_train_flatten / 255
    X_test = X_test_flatten / 255

    #转为one-hot 矩阵

    Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6)
    Y_test = tf_utils.convert_to_one_hot(Y_test_orig, 6)

    # 开始时间
    start_time = time.clock()
    # 开始训练
    parameters = model(X_train, Y_train, X_test, Y_test)
    # 结束时间
    end_time = time.clock()
    # 计算时差
    print("CPU的执行时间 = " + str(end_time - start_time) + " 秒")

    my_image1 = "2.jpg"

    fileName1 = r'E:\深度学习\第二课第三周编程作业\\' + my_image1

    image1 = mpimg.imread(fileName1)

    plt.imshow(image1)

    my_image1 = image1.reshape(1, 64 * 64 * 3).T

    my_image_prediction = tf_utils.predict(my_image1, parameters)

    print("预测结果: y = " + str(np.squeeze(my_image_prediction)))


