import numpy as np
import w2v_utils
import os


def cosine_similarity(u, v):
    """
    u与v的余弦相似度反映了u与v的相似程度

    参数：
        u -- 维度为(n,)的词向量
        v -- 维度为(n,)的词向量

    返回：
        cosine_similarity -- 由上面公式定义的u和v之间的余弦相似度。
    """


    #计算内积
    dot = np.dot(u, v)

    #计算u的二范数
    norm_u = np.sqrt(np.sum(np.power(u,2)))

    # 计算v的二范数
    norm_v = np.sqrt(np.sum(np.power(v, 2)))

    cosine_similarity = dot/(norm_u*norm_v)

    return cosine_similarity


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    解决“A与B相比就类似于C与____相比一样”之类的问题

    参数：
        word_a -- 一个字符串类型的词
        word_b -- 一个字符串类型的词
        word_c -- 一个字符串类型的词
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        best_word -- 满足(v_b - v_a) 最接近 (v_best_word - v_c) 的词
    """

    # 首先把单词都转为小写
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    v_a,v_b,v_c = word_to_vec_map[word_a],word_to_vec_map[word_b],word_to_vec_map[word_c]

    max_cosine_sim = -100
    v_best_word = None
    for i in word_to_vec_map.keys():
        if i in [word_a, word_b, word_c]:
            continue
        tmp_max = cosine_similarity(word_to_vec_map[i]-v_c,v_b-v_a)
        if tmp_max>max_cosine_sim :
            max_cosine_sim = tmp_max
            v_best_word = i

    return v_best_word


def neutralize(word, g, word_to_vec_map):

    """
    中和步
    通过将“word”投影到与偏置轴正交的空间上，消除了“word”的偏差。
    该函数确保“word”在性别的子空间中的值为0

    参数：
        word -- 待消除偏差的字符串
        g -- 维度为(50,)，对应于偏置轴（如性别）
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        e_debiased -- 消除了偏差的向量。
    """
    e = word_to_vec_map[word]

    #e_biascomponent = np.dot(e,g)/np.square(np.linalg.norm(g))*g
    e_biascomponent = np.dot(e, g) / np.square(np.sqrt(np.sum(np.power(g, 2)))) * g

    e_debiased = e - e_biascomponent

    return e_debiased


def equalize(pair, bias_axis, word_to_vec_map):
    """
    通过遵循上图中所描述的均衡方法来消除性别偏差。

    参数：
        pair -- 要消除性别偏差的词组，比如 ("actress", "actor")
        bias_axis -- 维度为(50,)，对应于偏置轴（如性别）
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        e_1 -- 第一个词的词向量
        e_2 -- 第二个词的词向量
    """
    # 将词转换为词向量
    word1,word2 = pair
    e_w1,e_w2 = word_to_vec_map[word1],word_to_vec_map[word2]
    # 计算除了bias方向的向量
    mu = (e_w1+e_w2)/2
    mu_B = np.dot(mu, bias_axis) / np.square(np.sqrt(np.sum(np.power(bias_axis, 2)))) * bias_axis
    mu_orth = mu - mu_B

    # 计算e_w1,e_w2 在bias方向上的分量
    e_w1_B = np.dot(e_w1, bias_axis) / np.square(np.sqrt(np.sum(np.power(bias_axis, 2)))) * bias_axis
    e_w2_B = np.dot(e_w2, bias_axis) / np.square(np.sqrt(np.sum(np.power(bias_axis, 2)))) * bias_axis

    # 调整e_w1B 与 e_w2B的偏置部分
    corrected_e_w1B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_orth)))) * np.divide(e_w1_B - mu_B,
                                                                                          np.abs(e_w1 - mu_orth - mu_B))
    corrected_e_w2B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_orth)))) * np.divide(e_w2_B - mu_B,
                                                                                          np.abs(e_w2 - mu_orth - mu_B))
    e_1 = corrected_e_w1B+mu_orth
    e_2 = corrected_e_w2B+mu_orth

    return e_1,e_2

if __name__ == '__main__':
    #加载已经训练好的词向量
    os.chdir(r"E:\深度学习\【中英】【吴恩达课后编程作业】Course 5 - 序列模型 - 第二周作业 - 词向量的运算与Emoji生成器")
    words, word_to_vec_map = w2v_utils.read_glove_vecs('data/glove.6B.50d.txt')
    #
    # # python 3.x
    # print(word_to_vec_map['hello'])
    g = word_to_vec_map['woman'] - word_to_vec_map['man']
    print("==========均衡校正前==========")
    print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
    print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
    e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
    print("\n==========均衡校正后==========")
    print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
    print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
