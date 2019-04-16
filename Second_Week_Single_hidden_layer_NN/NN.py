import numpy as np
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import matplotlib.pyplot as plt


class NN:
    def __init__(self):
        self.W1 = None
        self.W2 = None
        self.B1 = None
        self.B2 = None

    def build(self, hidden_num=4):
        '''
        :param X:输入训练数据
        :param Y: 训练数据标签
        '''

        input_size = X.shape[0]
        ouput_size = Y.shape[0]
        # 权重初始化
        self.W1 = np.random.randn(hidden_num, input_size)*0.01
        self.B1 = np.random.randn(hidden_num, 1)
        self.W2 = np.random.randn(ouput_size, hidden_num)*0.01
        self.B2 = np.random.randn(ouput_size, 1)

    def train(self, X, Y, epochs=200, lr=1.2):
        for i in range(epochs):
            # 正向传播
            Z1 = np.dot(self.W1, X) + self.B1
            A1 = np.tanh(Z1)
            Z2 = np.dot(self.W2, A1) + self.B2
            A2 = sigmoid(Z2)
            Train_Loss = -(np.dot(Y, np.log(A2.T)) + np.dot((1 - Y), np.log(1 - A2.T))) / X.shape[1]
            Train_ACC = np.sum(1 - np.logical_xor((A2 > 0.5).astype(np.uint8), Y)) / X.shape[1]
            print("第{}次训练，训练集Loss为：{},准确率为:{}".format(i, Train_Loss, Train_ACC))
            # 反向传播
            dZ2 = A2 - Y
            dW2 = np.dot(dZ2, A1.T) / X.shape[1]
            dB2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]
            dZ1 = np.dot(self.W2.T, dZ2) * (1 - np.power(A1, 2))
            dW1 = np.dot(dZ1, X.T) / X.shape[1]
            dB1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]
            self.W2 -= lr * dW2
            self.B2 -= lr * dB2
            self.W1 -= lr * dW1
            self.B1 -= lr * dB1

    def evolution(self, X):
        '''
        :param X:输入测试数据
        :return: 返回其类别
        '''
        Z1 = np.dot(self.W1, X) + self.B1
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.B2
        A2 = sigmoid(Z2)
        predictions = np.round(A2)
        return predictions


if __name__ == '__main__':
    X, Y = load_planar_dataset()
    model = NN()
    model.build()
    model.train(X, Y, 9000)
    plot_decision_boundary(lambda x: model.evolution(x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()
    predictions = model.evolution(X)
    Train_ACC = np.sum(1 - np.logical_xor((predictions > 0.5).astype(np.uint8), Y)) / X.shape[1]
    print('准确率: {}'.format(Train_ACC))
