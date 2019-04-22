import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets


def initialize_parameters_zeros(layer_num):
    parameters = {}
    for i in range(1, len(layer_num)):
        parameters['W' + str(i)] = np.zeros((layer_num[i], layer_num[i - 1]))
        parameters['b' + str(i)] = np.zeros((layer_num[i], 1))
    return parameters


def initialize_parameters_random(layer_num):
    np.random.seed(3)
    parameters = {}

    for i in range(1, len(layer_num)):
        parameters['W' + str(i)] = np.random.randn(layer_num[i], layer_num[i - 1]) *10
        parameters['b' + str(i)] = np.zeros((layer_num[i], 1))
    return parameters


def initialize_parameters_he(layer_num):
    parameters = {}
    for i in range(1, len(layer_num)):
        parameters['W' + str(i)] = np.random.randn(layer_num[i], layer_num[i - 1]) * np.sqrt(1 / layer_num[i - 1])
        parameters['b' + str(i)] = np.zeros((layer_num[i], 1))
    return parameters
