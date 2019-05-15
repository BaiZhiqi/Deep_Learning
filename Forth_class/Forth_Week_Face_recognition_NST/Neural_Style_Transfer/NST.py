import time
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow,show
from PIL import Image
import nst_utils
import numpy as np
import tensorflow as tf
from keras import backend as K

def compute_content_cost(a_C, a_G):
    """
    计算内容代价的函数

    参数：
        a_C -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像C的内容的激活值。
        a_G -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像G的内容的激活值。

    返回：
        J_content -- 实数，用上面的公式1计算的值。

    """
    m, n_H, n_W, n_C =  a_G.get_shape().as_list()
    # 对a_C与a_G从3维降到2维
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))*( 1/(4*n_H*n_W*n_C))

    return J_content
def gram_matrix(A):

    GA = tf.matmul(A, A, transpose_b=True)

    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    计算单隐藏层的风格损失

    参数：
        a_S -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像S的风格的激活值。
        a_G -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像G的风格的激活值。

    返回：
        J_style -- 实数，用上面的公式2计算的值。

    """
    # 第1步：从a_G中获取维度信息
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # 第2步，将a_S与a_G的维度重构为(n_C, n_H * n_W)
    a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style = tf.reduce_sum(tf.square(tf.subtract(GS,GG)))*( 1/(4*n_H*n_H*n_W*n_W*n_C*n_C))

    return J_style
def compute_style_cost(model, STYLE_LAYERS):
    """
      计算几个选定层的总体风格成本

      参数：
          model -- 加载了的tensorflow模型
          STYLE_LAYERS -- 字典，包含了：
                          - 我们希望从中提取风格的层的名称
                          - 每一层的系数（coeff）
      返回：
          J_style - tensor类型，实数，由公式(2)定义的成本计算方式来计算的值。

      """
    # 初始化所有的成本值
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        # 选择当前选定层的输出
        out = model[layer_name]

        #这部分没有看懂
        # 运行会话，将a_S设置为我们选择的隐藏层的激活值
        a_S = sess.run(out)

        # 将a_G设置为来自同一图层的隐藏层激活,这里a_G引用model[layer_name]，并且还没有计算，
        # 在后面的代码中，我们将图像G指定为模型输入，这样当我们运行会话时，
        # 这将是以图像G作为输入，从隐藏层中获取的激活值。
        a_G = out

        # 计算当前层的风格成本
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # 计算总风格成本，同时考虑到系数。
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    计算总成本

    参数：
        J_content -- 内容成本函数的输出
        J_style -- 风格成本函数的输出
        alpha -- 超参数，内容成本的权值
        beta -- 超参数，风格成本的权值

    """
    Cost = alpha * J_content + beta * J_style
    return Cost

def model_nn(sess, input_image, num_iterations = 200, is_print_info = True,
             is_plot = True, is_save_process_image = True,
             save_last_image_to = "output/generated_image.jpg"):
    #初始化全局变量
    sess.run(tf.global_variables_initializer())

    #运行带噪声的输入图像
    sess.run(model["input"].assign(input_image))

    for i in range(num_iterations):
        #运行最小化的目标：
        sess.run(train_step)

        #产生把数据输入模型后生成的图像
        generated_image = sess.run(model["input"])

        if is_print_info and i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("第 " + str(i) + "轮训练," +
                  "  总成本为:"+ str(Jt) +
                  "  内容成本为：" + str(Jc) +
                  "  风格成本为：" + str(Js))
        if is_save_process_image:
            nst_utils.save_image("output/" + str(i) + ".png", generated_image)

    nst_utils.save_image(save_last_image_to, generated_image)

    return generated_image


if __name__ == '__main__':

    tmp_dir = os.path.abspath(os.curdir)
    os.chdir(r"E:\深度学习\第四课第四周编程作业\Neural Style Transfer")
    #model = nst_utils.load_vgg_model("pretrained-model\imagenet-vgg-verydeep-19.mat")
    # 重设图
    tf.reset_default_graph()
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]
    # 第1步：创建交互会话
    sess = tf.InteractiveSession()

    # 第2步：加载内容图像(卢浮宫博物馆图片),并归一化图像
    content_image = scipy.misc.imread("images/louvre_small.jpg")
    content_image = nst_utils.reshape_and_normalize_image(content_image)

    # 第3步：加载风格图像(印象派的风格),并归一化图像
    style_image = scipy.misc.imread("images/monet.jpg")
    style_image = nst_utils.reshape_and_normalize_image(style_image)

    # 第4步：随机初始化生成的图像,通过在内容图像中添加随机噪声来产生噪声图像
    generated_image = nst_utils.generate_noise_image(content_image)
    imshow(generated_image[0])
    show()
    # 第5步：加载VGG16模型
    model = nst_utils.load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

    # 第6步：构建TensorFlow图：

    ##将内容图像作为VGG模型的输入。
    sess.run(model["input"].assign(content_image))

    ## 获取conv4_2层的输出
    out = model["conv4_2"]

    ## 将a_C设置为“conv4_2”隐藏层的激活值。
    a_C = sess.run(out)

    ## 将a_G设置为来自同一图层的隐藏层激活,这里a_G引用model["conv4_2"]，并且还没有计算，
    ## 在后面的代码中，我们将图像G指定为模型输入，这样当我们运行会话时，
    ## 这将是以图像G作为输入，从隐藏层中获取的激活值。
    a_G = out

    ## 计算内容成本
    J_content = compute_content_cost(a_C, a_G)

    ## 将风格图像作为VGG模型的输入
    sess.run(model["input"].assign(style_image))

    ## 计算风格成本
    J_style = compute_style_cost(model, STYLE_LAYERS)

    ## 计算总成本
    J = total_cost(J_content, J_style, alpha=10, beta=40)

    ## 定义优化器,设置学习率为2.0
    optimizer = tf.train.AdamOptimizer(2.0)

    ## 定义学习目标：最小化成本
    train_step = optimizer.minimize(J)

    # 开始时间
    start_time = time.clock()

    # 非GPU版本,约25-30min
    generated_image = model_nn(sess, generated_image)


    # 使用GPU，约1-2min
    # with tf.device("/gpu:0"):
    #     generated_image = model_nn(sess, generated_image)

    # 结束时间
    end_time = time.clock()

    # 计算时差
    minium = end_time - start_time

    print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium % 60)) + "秒")




