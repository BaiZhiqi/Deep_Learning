import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import reg_utils


def forward_propagation_drop(X, parameters, keep_prob):
    np.random.seed(1)
    i = 1
    W = [None]
    B = [None]
    cache = []
    a = X
    while 'W' + str(i) in parameters:
        W.append(parameters['W' + str(i)])
        B.append(parameters['b' + str(i)])
        i += 1
    # 前向传播开始
    for j in range(1, i - 1):
        z = np.dot(W[j], a) + B[j]
        a = reg_utils.relu(z)
        D = np.random.rand(a.shape[0], a.shape[1])
        D = D < keep_prob
        a = a * D
        a /= keep_prob
        cache.append(z)
        cache.append(D)
        cache.append(a)
        cache.append(W[j])
        cache.append(B[j])
    z = np.dot(W[i - 1], a) + B[i - 1]
    a = reg_utils.sigmoid(z)
    cache.append(z)
    cache.append(a)
    cache.append(W[i - 1])
    cache.append(B[i - 1])
    return a, cache


def backward_propagation_drop(X, Y, cache, keep_prob):
    Z = []
    D = []
    A = [X]
    W = []
    B = []
    m = X.shape[1]
    gradient = {}
    for i in range(0, len(cache) - 4, 5):
        Z.append(cache[i])
        D.append(cache[i + 1])
        A.append(cache[i + 2])
        W.append(cache[i + 3])
        B.append(cache[i + 4])
    layer_num = (len(cache) - 4) // 5
    Z.append(cache[-4])
    A.append(cache[-3])
    W.append(cache[-2])
    B.append(cache[-1])
    An = A.pop()
    dZ = An - Y
    dW = (1 / m)* np.dot(dZ, A[-1].T)
    dB = np.sum(dZ, axis=1, keepdims=True) / m
    gradient['dZ' + str(layer_num+1)] = dZ
    gradient['dW' + str(layer_num+1)] = dW
    gradient['db' + str(layer_num+1)] = dB
    for i in range(layer_num, 0, -1):
        An = A.pop()
        Dn = D.pop()
        dA = np.dot(W[i].T, dZ)
        dA = dA * Dn
        dA = dA / keep_prob
        dZ = np.multiply(dA, np.int64(An > 0))
        dW = 1. / m * np.dot(dZ, A[-1].T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        gradient['dA' + str(i)] = dA
        gradient['dZ' + str(i)] = dZ
        gradient['dW' + str(i)] = dW
        gradient['db' + str(i)] = db
    return gradient


def compute_cost_l2(A3, Y, parameters, lambd):
    Train_Loss = reg_utils.compute_cost(A3, Y)
    W = []
    i = 1
    while 'W' + str(i) in parameters:
        W.append(np.sum(np.square(parameters['W' + str(i)])))
        i += 1
    Total_Loss = (Train_Loss + lambd * np.sum(W) / (2 * Y.shape[1]))
    return Total_Loss


def backward_propagation_l2(X, Y, cache, lambd):
    Z = []
    A = [X]
    W = []
    B = []
    m = X.shape[1]
    gradient = {}
    for i in range(0, len(cache), 4):
        Z.append(cache[i])
        A.append(cache[i + 1])
        W.append(cache[i + 2])
        B.append(cache[i + 2])
    layer_num = len(cache) // 4
    An = A.pop()
    dZ = An - Y
    dW = np.dot(dZ, A[-1].T) / m + ((lambd * W[-1]) / m)
    dB = np.sum(dZ, axis=1, keepdims=True) / m
    gradient['dZ' + str(layer_num)] = dZ
    gradient['dW' + str(layer_num)] = dW
    gradient['db' + str(layer_num)] = dB
    for i in range(layer_num - 1, 0, -1):
        An = A.pop()
        dA = np.dot(W[i].T, dZ)
        dZ = np.multiply(dA, np.int64(An > 0))
        dW = 1. / m * np.dot(dZ, A[-1].T) + ((lambd * W[i-1]) / m)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        gradient['dA' + str(i)] = dA
        gradient['dZ' + str(i)] = dZ
        gradient['dW' + str(i)] = dW
        gradient['db' + str(i)] = db
    return gradient
