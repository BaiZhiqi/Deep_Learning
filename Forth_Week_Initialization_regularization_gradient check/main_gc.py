import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils  # 第一部分，初始化
import gc_utils
from Initialization import initialize_parameters_zeros,initialize_parameters_random,initialize_parameters_he
from testCases import gradient_check_n_test_case
def forward_propagation_n(X,Y,parameters):
    W1 = parameters['W1']
    B1 = parameters['b1']
    W2 = parameters['W2']
    B2 = parameters['b2']
    W3 = parameters['W3']
    B3 = parameters['b3']

    Z1 = np.dot(W1,X)+B1
    A1 = gc_utils.relu(Z1)

    Z2 = np.dot(W2,A1) + B2
    A2 = gc_utils.relu(Z2)

    Z3 = np.dot(W3, A2) + B3
    A3 = gc_utils.sigmoid(Z3)
    loss = -(np.dot(Y, np.log(A3.T)) + np.dot((1 - Y), np.log(1 - A3.T))) / X.shape[1]
    cache = (Z1, A1, W1, B1, Z2, A2, W2, B2, Z3, A3, W3, B3)
    return loss, cache
def backward_propagation_n(X,Y,cache):
    m = X.shape[1]
    (Z1, A1, W1, B1, Z2, A2, W2, B2, Z3, A3, W3, B3) = cache
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3,A2.T)/m
    dB3 = np.sum(dZ3, axis=1, keepdims=True)/m

    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(A2,np.int8(A2>0))
    dW2 = np.dot(dZ2,A1.T)/m
    dB2 = np.sum(dZ2, axis=1, keepdims=True)/m

    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(A1,np.int8(A1>0))
    dW1 = np.dot(dZ1,X.T)/m
    dB1 = np.sum(dZ1, axis=1, keepdims=True)/m

    gadient = {"dZ3":dZ3,"dW3":dW3,"db3":dB3,
               "dA2":dA2,"dZ2":dZ2,"dW2":dW2,"db2":dB2,
               "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": dB1}
    return gadient
def gradient_check_n(parameters,gradients,X,Y,epsilon=1e-7):
    parameters_values, keys = gc_utils.dictionary_to_vector(parameters)  # keys用不到
    grad = gc_utils.gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    for i in range(num_parameters):
        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon
        J_plus[i], cache = forward_propagation_n(X, Y, gc_utils.vector_to_dictionary(thetaplus))

        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[i][0] = thetaplus[i][0] - epsilon
        J_minus[i], cache = forward_propagation_n(X, Y, gc_utils.vector_to_dictionary(thetaplus))

        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    if difference < 1e-7:
        print("梯度检查：梯度正常!")
    else:
        print("梯度检查：梯度超出阈值!")

    return difference
if __name__ == '__main__':
    X,Y,parameters = gradient_check_n_test_case()
    loss,cache = forward_propagation_n(X,Y,parameters)
    gradients = backward_propagation_n(X,Y,cache)
    difference = gradient_check_n(parameters,gradients,X,Y)
    print(difference)