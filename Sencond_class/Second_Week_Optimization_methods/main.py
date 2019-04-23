# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import opt_utils
import testCases
plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
def update_parameters_with_gd(parameters,grads,learning_rate):
    l = len(parameters)//2
    for i in range(l):
        parameters["W"+str(i+1)] -=  learning_rate*grads["dW"+str(i+1)]
        parameters["b" + str(i + 1)] -= learning_rate * grads["db" + str(i + 1)]
    return parameters
def random_mini_batches(X,Y,mini_batch_size=64,seed=0):
    np.random.seed(seed)  # 指定随机种子
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    X_shuffle = X[:,permutation]
    Y_shuffle = Y[:,permutation].reshape((1,m))

    num_complete_minibatches = m // mini_batch_size
    for i in range(num_complete_minibatches):
        X_tmp = X_shuffle[:,i*mini_batch_size:(i+1)*mini_batch_size]
        Y_tmp = Y_shuffle[:,i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch = (X_tmp, Y_tmp)
        mini_batches.append(mini_batch)

    X_tmp = X_shuffle[:, (i+1) * mini_batch_size:]
    Y_tmp = Y_shuffle[:, (i+1) * mini_batch_size:]
    mini_batches.append((X_tmp,Y_tmp))
    return mini_batches
def initialize_velocity(parameters):
    L = len(parameters) // 2  # 神经网络的层数
    v = {}
    for i in range(L):
        v['dW'+str(i+1)] = np.zeros_like(parameters['W'+str(i+1)])
        v['db' + str(i + 1)] = np.zeros_like(parameters['b' + str(i + 1)])
    return v
def update_parameters_with_momentum(parameters,grads,v,beta,learning_rate):
    '''
    :param parameters: 神经网络的权重
    :param grads: 神经网路反向传播的梯度
    :param v: 动量
    :param beta: 超参数动量
    :param learning_rate:学习率
    :return:
    '''
    L = len(parameters) // 2
    for i in range(L):
        v['dW'+str(i+1)] = beta * v['dW'+str(i+1)] + (1-beta)*grads['dW'+str(i+1)]
        v['db' + str(i + 1)] = beta * v['db' + str(i + 1)] + (1 - beta) * grads['db' + str(i + 1)]
        parameters['W'+str(i+1)] = parameters['W'+str(i+1)] - learning_rate*v['dW'+str(i+1)]
        parameters['b' + str(i + 1)] = parameters['b' + str(i + 1)] - learning_rate * v['db' + str(i + 1)]
    return parameters,v
def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for i in range(L):
        v['dW' + str(i+1)] = np.zeros_like(parameters['W'+ str(i+1)])
        v['db' + str(i + 1)] = np.zeros_like(parameters['b' + str(i + 1)])
        s['dW' + str(i + 1)] = np.zeros_like(parameters['W' + str(i + 1)])
        s['db' + str(i + 1)] = np.zeros_like(parameters['b' + str(i + 1)])
    return v,s
def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
    '''
    :param parameters:神经网络的权重
    :param grads:神经网路反向传播的梯度
    :param v:momentum 动量
    :param s:RMSProp 动量
    :param t:第t个epoch
    :param learning_rate:学习率
    :param beta1:momentum超参数动量
    :param beta2:RMSProp超参数动量
    :param epsilon: 平滑
    :return:
    '''
    L = len(parameters) // 2
    v_correcte = {}
    s_correcte = {}
    for i in range(L):
        v['dW' + str(i + 1)] = (beta1 * v['dW' + str(i + 1)] + (1 - beta1) * grads['dW' + str(i + 1)])
        v['db' + str(i + 1)] = (beta1 * v['db' + str(i + 1)] + (1 - beta1) * grads['db' + str(i + 1)])

        v_correcte['dW' + str(i + 1)] = v['dW' + str(i + 1)]/(1-beta1**t)
        v_correcte['db' + str(i + 1)] = v['db' + str(i + 1)] / (1 - beta1 ** t)

        s['dW' + str(i + 1)] = (beta2 * s['dW' + str(i + 1)] + (1 - beta2) * (grads['dW' + str(i + 1)]**2))
        s['db' + str(i + 1)] = (beta2 * s['db' + str(i + 1)] + (1 - beta2) * (grads['db' + str(i + 1)]**2))

        s_correcte['dW' + str(i + 1)] =  s['dW' + str(i + 1)]/ (1 - beta2 ** t)
        s_correcte['db' + str(i + 1)] = s['db' + str(i + 1)] / (1 - beta2 ** t)

        parameters['W' + str(i + 1)] = parameters['W' + str(i + 1)] - learning_rate * (v_correcte['dW' + str(i + 1)]/(np.sqrt(s_correcte['dW' + str(i + 1)])+epsilon))
        parameters['b' + str(i + 1)] = parameters['b' + str(i + 1)] - learning_rate * (v_correcte['db' + str(i + 1)]/(np.sqrt(s_correcte['db' + str(i + 1)])+epsilon))
    return parameters,v,s
def model(X,Y,layer_dims,optimizer='adam',learning_rate=0.0007,beta1=0.9,beta2=0.999,epsilon=1e-8,minibatch_size=64,epochs=10000,print_cost=True,is_plot=True):
    '''
    :param X:输入图像
    :param Y:输入标签
    :param layer_dims:输入网络每层神经元的个数
    :param optimizer:选择优化器 包括'adam','gd','momentum'
    :param learning_rate:学习率
    :param beta1:momentum超参数动量
    :param beta2:RMSProp超参数动量
    :param epsilon:平滑
    :param minibatch_size:minibatch的大小
    :param epochs:循环多少次
    :param print_cost:是否输出Cost
    :param is_plot:是否显示可视化结果
    :return:参数
    '''
    costs = []
    t = 0
    seed = 10
    #初始化网络权重
    pass
    parameters = opt_utils.initialize_parameters(layer_dims)

    if optimizer=='adam':
        v,s = initialize_adam(parameters)
    elif optimizer=='momentum':
        v = initialize_velocity(parameters)
    elif optimizer=='gd':
        pass
    else:
        print('输入参数出错，程序退出。')
        exit(1)
    for i in range(epochs):
        seed += 1
        mini_batches = random_mini_batches(X, Y, minibatch_size,seed)
        for j in mini_batches:
            (minibatch_X, minibatch_Y) = j
            #前向传播
            a3, cache =  opt_utils.forward_propagation(minibatch_X,parameters)
            #计算Cost
            Loss = opt_utils.compute_cost(a3,minibatch_Y)
            #反向传播
            grads = opt_utils.backward_propagation(minibatch_X, minibatch_Y, cache)
            #更新参数
            if optimizer == 'adam':
                t += 1
                parameters, v, s = update_parameters_with_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
            elif optimizer == 'momentum':
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta1, learning_rate)
            else:
                parameters = update_parameters_with_gd(parameters,grads,learning_rate)
        if i % 100==0:
            costs.append(Loss)
            if print_cost and i % 1000 == 0:
                print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(Loss))
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()
    return parameters

if __name__ == '__main__':
    '''
    #测试gd优化梯度
    train_X, train_Y = opt_utils.load_dataset(is_plot=False)
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer="gd", is_plot=True)

    preditions = opt_utils.predict(train_X, train_Y, parameters)

    # 绘制分类图
    
    plt.title("Model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
    
    #测试momentum优化器
    train_X, train_Y = opt_utils.load_dataset(is_plot=False)
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer="momentum", is_plot=True)

    preditions = opt_utils.predict(train_X, train_Y, parameters)

    # 绘制分类图
    plt.title("Model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
    '''
    #测试adam优化器
    train_X, train_Y = opt_utils.load_dataset(is_plot=False)
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer="adam", is_plot=True)

    preditions = opt_utils.predict(train_X, train_Y, parameters)

    # 绘制分类图
    plt.title("Model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)

