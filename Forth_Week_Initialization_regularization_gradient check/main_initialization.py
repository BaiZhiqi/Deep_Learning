import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils  # 第一部分，初始化
from Initialization import initialize_parameters_zeros,initialize_parameters_random,initialize_parameters_he


plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he", is_polt=True):
    losses = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    # 选择初始化参数的类型
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
    for i in range(num_iterations):
        # 前向传播
        a3, cache = init_utils.forward_propagation(X,parameters)
        # 计算Loss
        loss = init_utils.compute_loss(a3,Y)

        # 反向传播
        gradients = init_utils.backward_propagation(X,Y,cache)
        #更新权重
        parameters = init_utils.update_parameters(parameters, gradients, learning_rate)
        if i % 1000 == 0:
            losses.append(loss)
            # 打印成本
            if print_cost:
                print("第" + str(i) + "次迭代，成本值为：" + str(loss))
    # 学习完毕，绘制成本曲线
    if is_polt:
        plt.plot(losses)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    # 返回学习完毕后的参数
    return parameters
if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot =  False)
    parameters = model(train_X,train_Y,initialization="random")
    print("训练集:")
    predictions_train = init_utils.predict(train_X, train_Y, parameters)
    print("测试集:")
    predictions_test = init_utils.predict(test_X, test_Y, parameters)

    plt.title("Model with large random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
