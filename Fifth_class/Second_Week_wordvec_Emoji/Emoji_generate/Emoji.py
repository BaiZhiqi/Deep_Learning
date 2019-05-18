import numpy as np
import emo_utils
import emoji
import matplotlib.pyplot as plt
import os
os.chdir(r"E:\深度学习\【中英】【吴恩达课后编程作业】Course 5 - 序列模型 - 第二周作业 - 词向量的运算与Emoji生成器")


def sentence_to_avg(sentence, word_to_vec_map):
    """
    将句子转换为单词列表，提取其GloVe向量，然后将其平均。

    参数：
        sentence -- 字符串类型，从X中获取的样本。
        word_to_vec_map -- 字典类型，单词映射到50维的向量的字典

    返回：
        avg -- 对句子的均值编码，维度为(50,)
    """
    words = sentence.lower().split()
    avg = np.zeros(50,)

    for i in words:
        avg +=word_to_vec_map[i]
    avg = avg/len(words)

    return avg


def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """
    在numpy中训练词向量模型。

    参数：
        X -- 输入的字符串类型的数据，维度为(m, 1)。
        Y -- 对应的标签，0-7的数组，维度为(m, 1)。
        word_to_vec_map -- 字典类型的单词到50维词向量的映射。
        learning_rate -- 学习率.
        num_iterations -- 迭代次数。

    返回：
        pred -- 预测的向量，维度为(m, 1)。
        W -- 权重参数，维度为(n_y, n_h)。
        b -- 偏置参数，维度为(n_y,)
    """
    np.random.seed(1)

    # 定义训练数量
    m = Y.shape[0]
    n_y = 5
    n_h = 50

    W = np.random.randn(n_y, n_h)/np.sqrt(n_h)
    b = np.zeros((n_y,))

    Y_oh = emo_utils.convert_to_one_hot(Y, C=n_y)
    for i in range(num_iterations):
        for j in range(m):
            avg = sentence_to_avg(X[j],word_to_vec_map)

            z = np.dot(W,avg)+b
            a = emo_utils.softmax(z)

            loss = -np.sum(np.dot(Y_oh[j] , np.log(a)))
            dz = a - Y_oh[j]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz

            W = W - learning_rate*dW
            b = b - learning_rate*db
        if i % 100 == 0:
            print("第{t}轮，损失为{cost}".format(t=i, cost=loss))
            pred = emo_utils.predict(X, Y, W, b, word_to_vec_map)
    return pred,W,b
if __name__ == '__main__':
    X_train, Y_train = emo_utils.read_csv('data/train_emoji.csv')
    X_test, Y_test = emo_utils.read_csv('data/test.csv')
    word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs('data/glove.6B.50d.txt')
    maxLen = len(max(X_train, key=len).split())
    Y_oh_train = emo_utils.convert_to_one_hot(Y_train, C=5)
    Y_oh_test = emo_utils.convert_to_one_hot(Y_test, C=5)

    print(X_train.shape)
    print(Y_train.shape)
    print(np.eye(5)[Y_train.reshape(-1)].shape)
    print(X_train[0])
    print(type(X_train))
    Y = np.asarray([5, 0, 0, 5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
    print(Y.shape)

    X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
                    'Lets go party and drinks', 'Congrats on the new job', 'Congratulations',
                    'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
                    'You totally deserve this prize', 'Let us go play football',
                    'Are you down for football this afternoon', 'Work hard play harder',
                    'It is suprising how people can be dumb sometimes',
                    'I am very disappointed', 'It is the best day in my life',
                    'I think I will end up alone', 'My life is so boring', 'Good job',
                    'Great so awesome'])
    pred, W, b = model(X_train, Y_train, word_to_vec_map)
    print("=====训练集====")
    pred_train = emo_utils.predict(X_train, Y_train, W, b, word_to_vec_map)
    print("=====测试集====")
    pred_test = emo_utils.predict(X_test, Y_test, W, b, word_to_vec_map)

    X_my_sentences = np.array(
        ["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "you are not happy"])
    Y_my_labels = np.array([[0], [0], [2], [1], [4], [3]])

    pred = emo_utils.predict(X_my_sentences, Y_my_labels, W, b, word_to_vec_map)
    emo_utils.print_predictions(X_my_sentences, pred)

    print(" \t {0} \t {1} \t {2} \t {3} \t {4}".format(emo_utils.label_to_emoji(0), emo_utils.label_to_emoji(1), \
                                                       emo_utils.label_to_emoji(2), emo_utils.label_to_emoji(3), \
                                                       emo_utils.label_to_emoji(4)))
    import pandas as pd

    print(pd.crosstab(Y_test, pred_test.reshape(56, ), rownames=['Actual'], colnames=['Predicted'], margins=True))
    emo_utils.plot_confusion_matrix(Y_test, pred_test)
