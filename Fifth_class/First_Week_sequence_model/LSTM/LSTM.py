import numpy as np
import rnn_utils

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    根据图4实现一个LSTM单元的前向传播。

    参数：
        xt -- 在时间步“t”输入的数据，维度为(n_x, m)
        a_prev -- 上一个时间步“t-1”的隐藏状态，维度为(n_a, m)
        c_prev -- 上一个时间步“t-1”的记忆状态，维度为(n_a, m)
        parameters -- 字典类型的变量，包含了：
                        Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                        bf -- 遗忘门的偏置，维度为(n_a, 1)
                        Wi -- 更新门的权值，维度为(n_a, n_a + n_x)
                        bi -- 更新门的偏置，维度为(n_a, 1)
                        Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                        bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                        Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                        bo -- 输出门的偏置，维度为(n_a, 1)
                        Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                        by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)
    返回：
        a_next -- 下一个隐藏状态，维度为(n_a, m)
        c_next -- 下一个记忆状态，维度为(n_a, m)
        yt_pred -- 在时间步“t”的预测，维度为(n_y, m)
        cache -- 包含了反向传播所需要的参数，包含了(a_next, c_next, a_prev, c_prev, xt, parameters)
    """



    # 从“parameters”中获取相关值
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    contact = np.vstack((a_prev,xt))
    #遗忘门
    Gf = rnn_utils.sigmoid(np.dot(Wf,contact)+bf)
    #更新门
    Gi = rnn_utils.sigmoid(np.dot(Wi,contact)+bi)
    #输出门
    Go = rnn_utils.sigmoid(np.dot(Wo,contact)+bo)

    # 更新单元
    tmp_ct = np.tanh(np.dot(Wc,contact)+bc)

    # 更新单元
    ct = np.multiply(Gi,tmp_ct) + np.multiply(Gf,c_prev)

    #输出

    a_next = np.multiply(Go,np.tanh(ct))

    #计算LSTM单元的预测值

    y_pre = rnn_utils.softmax(np.dot(Wy, a_next) + by)

    cache = (a_next, ct, a_prev, c_prev,Gf, Gi, tmp_ct, Go,  xt, parameters)
    return a_next,ct,y_pre,cache

def lstm_forward(x, a0, parameters):
    """
        实现LSTM单元组成的的循环神经网络

        参数：
            x -- 所有时间步的输入数据，维度为(n_x, m, T_x)
            a0 -- 初始化隐藏状态，维度为(n_a, m)
            parameters -- python字典，包含了以下参数：
                            Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                            bf -- 遗忘门的偏置，维度为(n_a, 1)
                            Wi -- 更新门的权值，维度为(n_a, n_a + n_x)
                            bi -- 更新门的偏置，维度为(n_a, 1)
                            Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                            bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                            Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                            bo -- 输出门的偏置，维度为(n_a, 1)
                            Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                            by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)

        返回：
            a -- 所有时间步的隐藏状态，维度为(n_a, m, T_x)
            y -- 所有时间步的预测值，维度为(n_y, m, T_x)
            caches -- 为反向传播的保存的元组，维度为（【列表类型】cache, x)）
        """
    caches = []
    n_x,m,n_T = x.shape
    n_y, n_a = parameters["Wy"].shape
    a = np.zeros([n_a,m,n_T])
    c = np.zeros([n_a,m,n_T])
    y_pres = np.zeros([n_y,m,n_T])

    a_next = a0
    c_next = np.zeros([n_a, m])
    for i in range(n_T):
        a_next, c_next, y_pre, cache = lstm_cell_forward(x[...,i], a_next, c_next, parameters)
        a[...,i] = a_next
        c[..., i] = c_next
        y_pres[...,i] = y_pre
        caches.append(cache)
    caches = (caches, x)
    return a,y_pres,c,caches


def lstm_cell_backward(da_next, dc_next, cache):
    """
    实现LSTM的单步反向传播

    参数：
        da_next -- 下一个隐藏状态的梯度，维度为(n_a, m)
        dc_next -- 下一个单元状态的梯度，维度为(n_a, m)
        cache -- 来自前向传播的一些参数

    返回：
        gradients -- 包含了梯度信息的字典：
                        dxt -- 输入数据的梯度，维度为(n_x, m)
                        da_prev -- 先前的隐藏状态的梯度，维度为(n_a, m)
                        dc_prev -- 前的记忆状态的梯度，维度为(n_a, m, T_x)
                        dWf -- 遗忘门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbf -- 遗忘门的偏置的梯度，维度为(n_a, 1)
                        dWi -- 更新门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbi -- 更新门的偏置的梯度，维度为(n_a, 1)
                        dWc -- 第一个“tanh”的权值的梯度，维度为(n_a, n_a + n_x)
                        dbc -- 第一个“tanh”的偏置的梯度，维度为(n_a, n_a + n_x)
                        dWo -- 输出门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbo -- 输出门的偏置的梯度，维度为(n_a, 1)
    """
    # 从cache中获取信息
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    # 获取xt与a_next的维度信息
    n_x, m = xt.shape
    n_a, m = a_next.shape

    #输出门的偏导数 因为dot和at以及Y有关。
    dot = da_next*np.tanh(c_next)*ot*(1-ot)

    #临时C的偏导数
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))

    #更新门的偏导数
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)

    #遗忘门的偏导数
    dft = (da_next*ot*(1-np.square(np.tanh(c_next)))+dc_next)*c_prev*ft*(1-ft)

    #计算权重更新
    contact = np.vstack((a_prev, xt))

    dWf = np.dot(dft,contact.T)
    dWi = np.dot(dit,contact.T)
    dWc = np.dot(dcct,contact.T)
    dWo = np.dot(dot,contact.T)

    dbf = np.sum(dft,axis=1,keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)
    #计算先前隐藏状态、先前记忆状态和输入的倒数

    da_prev = np.dot(parameters["Wf"][:, :n_a].T, dft) + np.dot(parameters["Wc"][:, :n_a].T, dcct) + np.dot(
        parameters["Wi"][:, :n_a].T, dit) + np.dot(parameters["Wo"][:, :n_a].T, dot)

    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next

    dxt = np.dot(parameters["Wf"][:, n_a:].T, dft) + np.dot(parameters["Wc"][:, n_a:].T, dcct) + np.dot(
        parameters["Wi"][:, n_a:].T, dit) + np.dot(parameters["Wo"][:, n_a:].T, dot)

    # 保存梯度信息到字典
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

def lstm_backward(da, caches):
    """
        实现LSTM网络的反向传播

        参数：
            da -- 关于隐藏状态的梯度，维度为(n_a, m, T_x)
            cachses -- 前向传播保存的信息

        返回：
            gradients -- 包含了梯度信息的字典：
                            dx -- 输入数据的梯度，维度为(n_x, m，T_x)
                            da0 -- 先前的隐藏状态的梯度，维度为(n_a, m)
                            dWf -- 遗忘门的权值的梯度，维度为(n_a, n_a + n_x)
                            dbf -- 遗忘门的偏置的梯度，维度为(n_a, 1)
                            dWi -- 更新门的权值的梯度，维度为(n_a, n_a + n_x)
                            dbi -- 更新门的偏置的梯度，维度为(n_a, 1)
                            dWc -- 第一个“tanh”的权值的梯度，维度为(n_a, n_a + n_x)
                            dbc -- 第一个“tanh”的偏置的梯度，维度为(n_a, n_a + n_x)
                            dWo -- 输出门的权值的梯度，维度为(n_a, n_a + n_x)
                            dbo -- 输出门的偏置的梯度，维度为(n_a, 1)

        """

    # 从caches中获取第一个cache（t=1）的值
    caches, x = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    # 获取da与x1的维度信息
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # 初始化梯度
    dx = np.zeros([n_x, m, T_x])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])
    dc_prevt = np.zeros([n_a, m])
    dWf = np.zeros([n_a, n_a + n_x])
    dWi = np.zeros([n_a, n_a + n_x])
    dWc = np.zeros([n_a, n_a + n_x])
    dWo = np.zeros([n_a, n_a + n_x])
    dbf = np.zeros([n_a, 1])
    dbi = np.zeros([n_a, 1])
    dbc = np.zeros([n_a, 1])
    dbo = np.zeros([n_a, 1])

    for i in reversed(range(T_x)):
        gradients = lstm_cell_backward(da[...,i],dc_prevt,caches[i])
        dx[...,i] = gradients["dxt"]
        dWf = dWf + gradients['dWf']
        dWi = dWi + gradients['dWi']
        dWc = dWc + gradients['dWc']
        dWo = dWo + gradients['dWo']
        dbf = dbf + gradients['dbf']
        dbi = dbi + gradients['dbi']
        dbc = dbc + gradients['dbc']
        dbo = dbo + gradients['dbo']

        dc_prevt = gradients["dc_prev"]
    da0 = gradients['da_prev']
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients


if __name__ == '__main__':
    np.random.seed(1)
    x = np.random.randn(3, 10, 7)
    a0 = np.random.randn(5, 10)
    Wf = np.random.randn(5, 5 + 3)
    bf = np.random.randn(5, 1)
    Wi = np.random.randn(5, 5 + 3)
    bi = np.random.randn(5, 1)
    Wo = np.random.randn(5, 5 + 3)
    bo = np.random.randn(5, 1)
    Wc = np.random.randn(5, 5 + 3)
    bc = np.random.randn(5, 1)
    Wy = np.random.randn(1,5)
    by = np.random.randn(1, 1)
    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a, y, c, caches = lstm_forward(x, a0, parameters)

    da = np.random.randn(5, 10, 4)
    gradients = lstm_backward(da, caches)

    print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
    print("gradients[\"dx\"].shape =", gradients["dx"].shape)
    print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
    print("gradients[\"da0\"].shape =", gradients["da0"].shape)
    print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
    print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
    print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
    print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
    print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
    print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
    print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
    print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
    print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
    print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
    print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
    print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
    print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
    print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
    print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
    print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)


