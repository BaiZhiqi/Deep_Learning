import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


def zero_pad(X, pad):
    '''
    输入一个维度为4的矩阵，其中第一维表示输入头像张数，第二三维度表示宽和高，第四维表示通道数，
    需要pad的就是二三维
    '''
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    return X_pad


def conv2d_single_stage(area, W, b):
    '''
    :param area: 输入卷积核所对应的一个区域
    :param W: 权重
    :param b: 偏移
    '''

    Z = np.multiply(W, area) + b
    Z = np.sum(Z)
    return Z


def conv2d_forward(A_prev, W, b, pad, stride):
    '''
    :param A_prev:上层传入的激活矩阵
    :param W: 本层权重
    :param b: 偏移
    :param pad: 填充
    :param stride: 步长
    '''
    (num, height, width, channel) = A_prev.shape
    (f1, f2, channel, channels) = W.shape

    N_w = int((width + 2 * pad - f1) / stride + 1)
    N_h = int((height + 2 * pad - f2) / stride + 1)

    Z = np.zeros((num, N_h, N_w, channels))

    A = zero_pad(A_prev, pad)
    try:
        for i in range(num):
            tmp_A = A[i]
            for j in range(N_w):
                for k in range(N_h):
                    for l in range(channels):
                        width_start = j * stride
                        width_end = j * stride + f1
                        height_start = k * stride
                        height_end = k * stride + f2
                        Z[i, k, j, l] = conv2d_single_stage(tmp_A[height_start:height_end, width_start:width_end, :],
                                                            W[..., l], b[0, 0, 0, l])
                        # A[i, h, w, c] = activation(Z[i, h, w, c])
    except:
        print('卷积边界超出')
        return
    cache = (A_prev, W, b, pad, stride)
    return Z, cache


def pooling_forward(A_prev, f, stride, mode="max"):
    '''
    :param A_prev: 前层激活矩阵
    :param f: 池化的大小
    :param stride: 池化的步长
    :param mode: 模式选择，可以使最大池化或者平均池化
    '''
    (num, height, width, channel) = A_prev.shape

    N_w = int((width - f) / stride + 1)
    N_h = int((height - f) / stride + 1)

    Z = np.zeros((num, N_h, N_w, channel))

    try:
        for i in range(num):
            tmp_A = A_prev[i]
            for j in range(N_w):
                for k in range(N_h):
                    for l in range(channel):
                        width_start = j * stride
                        width_end = j * stride + f
                        height_start = k * stride
                        height_end = k * stride + f
                        if mode == 'max':
                            Z[i, k, j, l] = np.max(tmp_A[height_start:height_end, width_start:width_end, l])
                        elif mode == 'average':
                            Z[i, k, j, l] = np.mean(tmp_A[height_start:height_end, width_start:width_end, l])
                            # A[i, h, w, c] = activation(Z[i, h, w, c])
    except:
        print('池化边界超出')
        return
    cache = (A_prev, f, stride)
    return Z, cache


def conv2d_backward(dZ, cache):
    '''
    实现卷积的反向传播
    :param dZ:卷积层的输出Z的 梯度，维度为(m, n_H, n_W, n_C)
    :param cache:卷积的参数
    '''
    (A_prev, W, b, pad, stride) = cache
    (m, n_H, n_W, n_C) = dZ.shape
    (f1, f2, channel, channels) = W.shape
    dA_final = np.zeros(A_prev.shape)
    A = zero_pad(A_prev, pad)

    dA = np.zeros(A.shape)
    dW = np.zeros(W.shape)
    db = np.zeros((1, 1, 1, channels))

    for i in range(m):
        dA_tmp = dA[i]
        A_tmp = A[i]
        for j in range(n_W):
            for k in range(n_H):
                for l in range(channels):
                    width_start = j * stride
                    width_end = j * stride + f2
                    height_start = k * stride
                    height_end = k * stride + f1
                    dA_tmp[height_start:height_end, width_start:width_end, :] += np.multiply(dZ[i, k, j, l], W[..., l])
                    dW[..., l] += np.multiply(A_tmp[height_start:height_end, width_start:width_end, :], dZ[i, k, j, l])
                    db[..., l] += dZ[i, k, j, l]
        dA_final[i, ...] = dA_tmp[pad:-pad, pad:-pad, :]
    return (dA_final, dW, db)


def create_mask_from_window(x):
    '''
    返回大小相同的mask掩模，其中标记着最大值的位置
    :param x:输入的区域
    '''
    mask = x == np.max(x)
    return mask


def distribute_value(dz, shape):
    a = dz / (shape[0] * shape[1])
    result = np.ones(shape) * a
    return result


def pooling_backward(dZ, cache,mode='max'):
    (A_prev, f,stride) = cache

    dA = np.zeros_like(A_prev)

    (m, n_H, n_W, n_C) = dZ.shape

    for i in range(m):
        A_tmp = A_prev[i]
        for j in range(n_W):
            for k in range(n_H):
                for l in range(n_C):
                    width_start = j * stride
                    width_end = j * stride + f
                    height_start = k * stride
                    height_end = k * stride + f
                    if mode=='max':
                        dA[i,height_start:height_end, width_start:width_end, l] += np.multiply(dZ[i,k,j,l],create_mask_from_window(A_tmp[height_start:height_end, width_start:width_end, l]))
                    elif mode == 'average':
                        dA[i,height_start:height_end, width_start:width_end, l] += distribute_value(dZ[i, k, j, l],(f,f))

    return dA


if __name__ == '__main__':
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride": 1, "f": 2}
    A, cache = pooling_forward(A_prev, 2,1)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = pooling_backward(dA, cache, mode="max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])
    print()
    dA_prev = pooling_backward(dA, cache, mode="average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])

