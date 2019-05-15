from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

#------------用于绘制模型细节，可选--------------#
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#------------------------------------------------#

K.set_image_data_format('channels_first')

import time
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import fr_utils
from inception_blocks_v2 import *
import sys
np.set_printoptions(threshold=sys.maxsize)
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    根据公式（4）实现三元组损失函数

    参数：
        y_true -- true标签，当你在Keras里定义了一个损失函数的时候需要它，但是这里不需要。
        y_pred -- 列表类型，包含了如下参数：
            anchor -- 给定的“anchor”图像的编码，维度为(None,128)
            positive -- “positive”图像的编码，维度为(None,128)
            negative -- “negative”图像的编码，维度为(None,128)
        alpha -- 超参数，阈值

    返回：
        loss -- 实数，损失的值
    """
    anchor,positive,negative = y_pred[0],y_pred[1],y_pred[2]

    # 第一步：计算"anchor" 与 "positive"之间编码的距离，这里需要使用axis=-1

    list_anchor_positive = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)

    # 第二步：计算"anchor" 与 "negative"之间编码的距离，这里需要使用axis=-1

    list_anchor_negtive = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)

    #第三步：使list_anchor_positive比list_anchor_negtive+alpha更小

    basic_loss = tf.add(tf.subtract(list_anchor_positive , list_anchor_negtive) , alpha)

    #取basic_loss和0更大的那部分

    loss = tf.maximum(basic_loss, 0)

    #计算cost
    cost = tf.reduce_sum(loss)
    return cost


def verify(image_path, identity, database, model):
    """
    对“identity”与“image_path”的编码进行验证。

    参数：
        image_path -- 摄像头的图片。
        identity -- 字符类型，想要验证的人的名字。
        database -- 字典类型，包含了成员的名字信息与对应的编码。
        model -- 在Keras的模型的实例。

    返回：
        dist -- 摄像头的图片与数据库中的图片的编码的差距。
        is_open_door -- boolean,是否该开门。
    """
    encoding = fr_utils.img_to_encoding(image_path, model)

    #求距离的二范数
    dist = np.linalg.norm(encoding - database[identity])
    if dist<0.7:
        print("欢迎 " + str(identity) + "回家！")
        is_door_open = True
    else:
        print("经验证，您与" + str(identity) + "不符！")
        is_door_open = False
    return dist,is_door_open
def who_is_it(image_path, database,model):
    """
    根据指定的图片来进行人脸识别

    参数：
        images_path -- 图像地址
        database -- 包含了名字与编码的字典
        model -- 在Keras中的模型的实例。

    返回：
        min_dist -- 在数据库中与指定图像最相近的编码。
        identity -- 字符串类型，与min_dist编码相对应的名字。
    """
    min_dist  = 100
    result = {}
    encoding = fr_utils.img_to_encoding(image_path, model)
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if min_dist>dist:
            min_dist = dist
            result["name"] = name
            result["dist"] = dist
    if result["dist"]<0.7:
        print("姓名" + result["name"] + "  差距：" + str(result["dist"]))

    else:
        print("抱歉，您的信息不在数据库中。")
    return result

if __name__ == '__main__':
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    #编译模型
    FRmodel.compile(optimizer = 'adam',loss = triplet_loss,metrics=['accuracy'])
    #加载权重
    fr_utils.load_weights_from_FaceNet(FRmodel)

    database = {}
    tmp_dir = os.path.abspath(os.curdir)
    os.chdir(r"E:\深度学习\第四课第四周编程作业\Face Recognition\images")
    database["danielle"] = fr_utils.img_to_encoding("danielle.png", FRmodel)
    database["younes"] = fr_utils.img_to_encoding("younes.jpg", FRmodel)
    database["tian"] = fr_utils.img_to_encoding("tian.jpg", FRmodel)
    database["andrew"] = fr_utils.img_to_encoding("andrew.jpg", FRmodel)
    database["kian"] = fr_utils.img_to_encoding("kian.jpg", FRmodel)
    database["dan"] = fr_utils.img_to_encoding("dan.jpg", FRmodel)
    database["sebastiano"] = fr_utils.img_to_encoding("sebastiano.jpg", FRmodel)
    database["bertrand"] = fr_utils.img_to_encoding("bertrand.jpg", FRmodel)
    database["kevin"] = fr_utils.img_to_encoding("kevin.jpg", FRmodel)
    database["felix"] = fr_utils.img_to_encoding("felix.jpg", FRmodel)
    database["benoit"] = fr_utils.img_to_encoding("benoit.jpg", FRmodel)
    database["arnaud"] = fr_utils.img_to_encoding("arnaud.jpg", FRmodel)
    tmp = verify("camera_0.jpg", "younes", database, FRmodel)
    tmp = verify("camera_2.jpg", "kian", database, FRmodel)
    tmp = who_is_it("camera_0.jpg", database, FRmodel)
    os.chdir(tmp_dir)