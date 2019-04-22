import numpy as np
import matplotlib.pyplot as plt
import testCases_v2
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
from dnn_app_utils_v2 import load_data


class DNN:
    def __init__(self):
        self.W = {}
        self.B = {}
        self.A = {}
        self.Z = {}
        self.dA = {}
        self.dW = {}
        self.dB = {}

    def build(self, layers_dims):
        '''
        :param layers_dims:各项权重的初始化
        '''
        np.random.seed(3)
        for i in range(1, len(layers_dims)):
            self.W[i] = np.random.randn(layers_dims[i], layers_dims[i - 1]) / np.sqrt(layers_dims[i - 1])
            self.B[i] = np.zeros((layers_dims[i], 1))

    def forward(self, i, A, activation):
        Z = np.dot(self.W[i], A) + self.B[i]
        if activation == 'relu':
            A, cache = relu(Z)
        else:
            A, cache = sigmoid(Z)
        return A, cache

    def back(self, i, dA, activation):

        if activation == 'relu':
            dZ = relu_backward(dA, self.Z[i])
        else:
            dZ = sigmoid_backward(dA, self.Z[i])
        dW = np.dot(dZ, self.A[i - 1].T) / self.A[i].shape[1]
        dB = np.sum(dZ, axis=1, keepdims=True) / self.A[i].shape[1]
        dA = np.dot(self.W[i].T, dZ)
        return dA, dW, dB

    def train(self, length, X, Y, TestX, TestY, lr=0.0075,epochs=2000):
        '''
        :param length:输入网络的深度
        '''
        self.A[0] = X

        for epoch in range(epochs):
            A = self.A[0]
            for i in range(1, length - 1):
                A, Z = self.forward(i, A, 'relu')
                self.A[i] = A
                self.Z[i] = Z
            A, Z = self.forward(length - 1, A, 'sigmoid')
            self.A[length - 1] = A
            self.Z[length - 1] = Z

            Train_Loss = -(np.dot(Y, np.log(A.T)) + np.dot((1 - Y), np.log(1 - A.T))) / X.shape[1]
            Train_ACC = np.sum(1 - np.logical_xor((A > 0.5).astype(np.uint8), Y)) / X.shape[1]
            print("第{}次训练，训练集Loss为：{},准确率为:{}".format(epoch, Train_Loss, Train_ACC))
            dAL = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))

            self.dA[length - 1], self.dW[length - 1], self.dB[length - 1] = self.back(length - 1, dAL, 'sigmoid')
            dA, dW, dB = self.dA[length - 1], self.dW[length - 1], self.dB[length - 1]
            for i in range(length - 2, 0, -1):
                dA, dW, dB = self.back(i, dA, 'relu')
                self.dA[i], self.dW[i], self.dB[i] = dA, dW, dB
            for i in range(1, length):
                self.W[i] -= lr * self.dW[i]
                self.B[i] -= lr * self.dB[i]
            self.evolution(length, TestX, TestY)

    def evolution(self, length, X, Y):
        A = X
        for i in range(1, length-1):
            A, Z = self.forward(i, A, 'relu')
        A, Z = self.forward(length-1, A, 'sigmoid')
        Test_Loss = -(np.dot(Y, np.log(A.T)) + np.dot((1 - Y), np.log(1 - A.T))) / X.shape[1]
        Test_ACC = np.sum(1 - np.logical_xor((A > 0.5).astype(np.uint8), Y)) / X.shape[1]
        print("测试集Loss为：{},准确率为:{}".format(Test_Loss, Test_ACC))


if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()
    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255
    train_y = train_set_y
    test_x = test_x_flatten / 255
    test_y = test_set_y

    model = DNN()
    layers_dims = [12288, 20, 7, 5, 1]
    model.build(layers_dims)
    model.train(len(layers_dims), train_x, train_y, test_x, test_y)
