import numpy as np
from math import log as log
from math import exp as exp
from load_datasets import load_data


def train(train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, batch_size, epoch, lr):
    # 处理数据，将训练图像展开，通过前向和反向来训练
    # 其中，train_set_x_orig为训练集图像，train_set_y_orig为训练集标签，test_set_x_orig为验证集图像，test_set_y_orig为验证集标签，epoch为迭代次数，lr为学习率
    w = np.zeros((1, np.shape(train_set_x_orig)[1]))  # 初始权重
    b = 0
    for i in range(0, epoch):
        # 训练过程
        '''
        if i + batch_size <len(train_set_y_orig):
            train_set_x_orig = train_set_x_orig[i:i+batch_size]
            train_set_y_orig = train_set_y_orig[:,i:i+batch_size]
        else:
            train_set_x_orig = train_set_x_orig[:,i:]
            train_set_y_orig = train_set_y_orig[:,i:]
        '''
        train_loss, train_acc, train_set_y_pred = Forward(train_set_x_orig, train_set_y_orig, w, b)
        # 反向传播
        w, b = Back(train_set_x_orig, train_set_y_orig, train_set_y_pred, w, b, lr)
        # 验证
        val_loss, val_acc,test_set_y_pred = Forward(test_set_x_orig, test_set_y_orig, w, b)
        print(r'第 {} epoch，train_loss = {},train_acc = {},val_loss = {},val_acc= {} '.format(i, train_loss, train_acc, val_loss,
                                                                              val_acc))



def Forward(train_set_x_orig, train_set_y_orig, w, b):
    # 根据w和b计算预测的类别，并计算loss和acc
    tmp = np.dot(w, train_set_x_orig.T) + b
    # 激活函数
    train_set_y_pred = 1 / (1 + np.exp(-tmp))
    loss = np.sum(-(np.dot(train_set_y_orig , np.log(train_set_y_pred).T) + np.dot((1 - train_set_y_orig) , np.log(1 - train_set_y_pred).T)) / len(
        train_set_x_orig))
    acc = np.sum(np.logical_or(train_set_y_orig, (train_set_y_pred>=0.5).astype(np.uint8))) / np.shape(train_set_y_orig)[1]
    return loss, acc, train_set_y_pred

def Back(train_set_x_orig, train_set_y_orig, train_set_y_pred, w, b, lr):
    dz = train_set_y_pred - train_set_y_orig
    dw = np.dot(dz, train_set_x_orig)
    db = dz
    w = (w - lr * dw) / len(train_set_x_orig)
    b = np.sum(b - lr * db) / len(train_set_x_orig)
    return w, b


def main():
    batch_size = 5
    epochs = 1000
    lr = 0.009
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    train_shape = np.shape(train_set_x_orig)
    test_shape = np.shape(test_set_x_orig)
    train_set_x_orig = train_set_x_orig.reshape((train_shape[0], train_shape[1] * train_shape[2] * train_shape[3]))/255
    test_set_x_orig = test_set_x_orig.reshape((test_shape[0], test_shape[1] * test_shape[2] * test_shape[3]))/255
    train(train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, batch_size, epochs, lr)


main()
