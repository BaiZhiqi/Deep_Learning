import numpy as np
np.random.seed(0)
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import os
np.random.seed(1)
import emo_utils
from keras.initializers import glorot_uniform


def sentences_to_indices(X, word_to_index, max_len):
    """
    输入的是X（字符串类型的句子的数组），再转化为对应的句子列表，
    输出的是能够让Embedding()函数接受的列表或矩阵（参见图4）。

    参数：
        X -- 句子数组，维度为(m, 1)
        word_to_index -- 字典类型的单词到索引的映射
        max_len -- 最大句子的长度，数据集中所有的句子的长度都不会超过它。

    返回：
        X_indices -- 对应于X中的单词索引数组，维度为(m, max_len)
    """
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentences_words = X[i].lower().split()

        j = 0
        for k in sentences_words:
            X_indices[i,j] = word_to_index[k]
            j+=1
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    创建Keras Embedding()层，加载已经训练好了的50维GloVe向量

    参数：
        word_to_vec_map -- 字典类型的单词与词嵌入的映射
        word_to_index -- 字典类型的单词到词汇表（400,001个单词）的索引的映射。

    返回：
        embedding_layer() -- 训练好了的Keras的实体层。
    """
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))
    #这里生成一个每一行是对应行号的词向量的词矩阵
    for word,index in  word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    # 构建embedding层。
    embedding_layer.build((None,))

    # 将嵌入层的权重设置为嵌入矩阵。
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    实现Emojify-V2模型的计算图

    参数：
        input_shape -- 输入的维度，通常是(max_len,)
        word_to_vec_map -- 字典类型的单词与词嵌入的映射。
        word_to_index -- 字典类型的单词到词汇表（400,001个单词）的索引的映射。

    返回：
        model -- Keras模型实体
    """
    sentence_indices = Input(input_shape, dtype='int32')

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    embeddings = embedding_layer(sentence_indices)

    X = LSTM(128,return_sequences=True)(embeddings)

    X = Dropout(0,5)(X)

    # return_sequences表示输出一个序列还是时间序列最后一个的结果。True 为输出全部时间序列结果 False为输出时间序列最后一个的结果

    X = LSTM(128, return_sequences=False)(X)

    X = Dropout(0.5)(X)

    X = Dense(5)(X)

    X = Activation('softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)

    return model
if __name__ == '__main__':
    os.chdir(r"E:\深度学习\【中英】【吴恩达课后编程作业】Course 5 - 序列模型 - 第二周作业 - 词向量的运算与Emoji生成器")
    word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs('data/glove.6B.50d.txt')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])
    max_Len = 10
    model = Emojify_V2((max_Len,), word_to_vec_map, word_to_index)
    model.summary()
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    X_train, Y_train = emo_utils.read_csv('data/train_emoji.csv')
    X_test, Y_test = emo_utils.read_csv('data/test.csv')
    X_train_indices = sentences_to_indices(X_train, word_to_index, max_Len)
    Y_train_oh = emo_utils.convert_to_one_hot(Y_train, C=5)
    model.fit(X_train_indices,Y_train_oh,epochs=200,batch_size=32,shuffle=True)


    C = 5

    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=max_Len)
    Y_test_oh = emo_utils.convert_to_one_hot(Y_test, C=5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)

    print("Test accuracy = ", acc)

    y_test_oh = np.eye(C)[Y_test.reshape(-1)]
    X_test_indices = sentences_to_indices(X_test, word_to_index, max_Len)
    pred = model.predict(X_test_indices)
    for i in range(len(X_test)):

        num = np.argmax(pred[i])
        if (num != Y_test[i]):
            print('正确表情：' + emo_utils.label_to_emoji(Y_test[i]) + '   预测结果： ' + X_test[i] + emo_utils.label_to_emoji(
                num).strip())

    x_test = np.array(['you are a so beautiful girl'])
    X_test_indices = sentences_to_indices(x_test, word_to_index, max_Len)
    print(x_test[0] + ' ' + emo_utils.label_to_emoji(np.argmax(model.predict(X_test_indices))))
