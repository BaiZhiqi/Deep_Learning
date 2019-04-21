import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import reg_utils
import regulation
from Initialization import initialize_parameters_zeros, initialize_parameters_random, initialize_parameters_he

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, is_plot=True, lambd=0, keep_prob=1):
    losses = []
    layers_dims = [X.shape[0], 20, 3, 1]

    # 选择初始化参数的类型
    parameters = reg_utils.initialize_parameters(layers_dims)
    for i in range(num_iterations):
        # 前向传播
        if keep_prob == 1:
            A3, cache = reg_utils.forward_propagation(X, parameters)
        else:
            A3, cache = regulation.forward_propagation_drop(X, parameters, keep_prob)

        # 计算Loss
        if lambd == 0:
            loss = reg_utils.compute_cost(A3, Y)
        else:
            loss = regulation.compute_cost_l2(A3, Y, parameters, lambd)

        # 反向传播
        if keep_prob == 1 and lambd == 0:
            gradients = reg_utils.backward_propagation(X, Y, cache)
        elif keep_prob != 1:
            gradients = regulation.backward_propagation_drop(X, Y, cache, keep_prob)
        elif lambd != 0:
            gradients = regulation.backward_propagation_l2(X, Y, cache, lambd)
        # 更新权重
        parameters = reg_utils.update_parameters(parameters, gradients, learning_rate)
        if i % 1000 == 0:
            losses.append(loss)
            # 打印成本
            if (print_cost and i % 10000 == 0):
                print("第" + str(i) + "次迭代，成本值为：" + str(loss))
    # 学习完毕，绘制成本曲线
    if is_plot:
        plt.plot(losses)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    # 返回学习完毕后的参数
    return parameters


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=False)
    # 测试不加正则化的模型

    parameters = model(train_X, train_Y, is_plot=True)
    print("训练集:")
    predictions_train = reg_utils.predict(train_X, train_Y, parameters)
    print("测试集:")
    predictions_test = reg_utils.predict(test_X, test_Y, parameters)
    plt.title("Model with L2-regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

    #测试增加了L2正则项的模型
    parameters = model(train_X, train_Y, lambd=0.7, is_plot=True)
    print("使用正则化，训练集:")
    predictions_train = reg_utils.predict(train_X, train_Y, parameters)
    print("使用正则化，测试集:")
    predictions_test = reg_utils.predict(test_X, test_Y, parameters)

    plt.title("Model with L2-regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

    parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3, is_plot=True)

    print("使用随机删除节点，训练集:")
    predictions_train = reg_utils.predict(train_X, train_Y, parameters)
    print("使用随机删除节点，测试集:")
    reg_utils.predictions_test = reg_utils.predict(test_X, test_Y, parameters)

    plt.title("Model with dropout")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
