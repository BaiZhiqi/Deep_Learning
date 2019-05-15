#coding=utf-8

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model

from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

import yolo_utils


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    '''
    将低置信度的去掉
    :param box_confidence:tensor类型，维度为（19,19,5,1）,包含19x19单元格中每个单元格预测的5个锚框中的所有的锚框的pc （一些对象的置信概率）。
    :param boxes: ensor类型，维度为(19,19,5,4)，包含了所有的锚框的（px,py,ph,pw ）。
    :param box_class_probs: tensor类型，维度为(19,19,5,80)，包含了所有单元格中所有锚框的所有对象( c1,c2,c3，···，c80 )检测的概率。
    :param threshold:实数，阈值，如果分类预测的概率高于它，那么这个分类预测的概率就会被保留。
    :return
    scores - tensor 类型，维度为(None,)，包含了保留了的锚框的分类概率。
    boxes - tensor 类型，维度为(None,4)，包含了保留了的锚框的(b_x, b_y, b_h, b_w)
    classess - tensor 类型，维度为(None,)，包含了保留了的锚框的索引
    '''

    # 对于5个锚点，每一个锚点有一个最大概率类别，以及所属类别的概率，所以我们首先要计算每一类的概率，然后找到最大概率的类别以及其概率，最后去掉不满足条件的概率。
    # 计算每一个框的得分，因为得分要乘以当前块是否存在目标的概率
    box_scores = box_confidence * box_class_probs

    # 找到最大锚所在的维度以及其概率

    box_class = K.argmax(box_scores, axis=-1)
    box_class_score = K.max(box_scores, axis=-1)

    # 去掉低得分部分

    mask = box_class_score >= 0.6

    # 对结果进行mask

    scores = tf.boolean_mask(box_class_score, mask)
    boxes = tf.boolean_mask(boxes, mask)
    classes = tf.boolean_mask(box_class, mask)

    return scores, boxes, classes


def IoU(box1, box2):
    '''
    计算交并比
    :param box1:第一个锚框，元组类型，（x1,y1,x2,y2）x1最小值 y1最小值
    :param box2:第二个锚框，元组类型，（x1,y1,x2,y2）x1最小值 y1最小值
    :return:实数，IoU结果
    '''
    # 计算交集
    length = np.minimum(box1[2], box2[2]) - np.maximum(box1[0], box2[0])
    width = np.minimum(box1[3], box2[3]) - np.maximum(box1[1], box2[1])
    inter_area = length * width

    # 计算并集

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    #计算交并比
    iou  = inter_area/union_area

    return iou
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):



    '''
    实现非极大值抑制（NMS）
    :param scores: tensor类型，维度为(None,)，yolo_filter_boxes()的输出
    :param boxes:  tensor类型，维度为(None,4)，yolo_filter_boxes()的输出，已缩放到图像大小
    :param scores: tensor类型，维度为(None,)，yolo_filter_boxes()的输出。
    :param max_boxes:整数，预测的锚框数量的最大值，也就是一张图中，我们可以标记处的最大的框的数量
    :param iou_threshold:实数，交并比
    :return:
    scores - tensor类型，维度为(,None)，每个锚框的预测的可能值
    boxes - tensor类型，维度为(4,None)，预测的锚框的坐标
    classes - tensor类型，维度为(,None)，每个锚框的预测的分类
    '''
    max_boxes_tensor = K.variable(max_boxes, dtype="int32")  # 用于tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # 初始化变量max_boxes_tensor

    # 使用使用tf.image.non_max_suppression()来获取与我们保留的框相对应的索引列表
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    # 使用K.gather()来选择保留的锚框 根据nums_indices的索引将score的第几个实例进行保留
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes
def yolo_eval(yolo_outputs, image_shape=(720.,1280.),
              max_boxes=10, score_threshold=0.6,iou_threshold=0.5):
    """
    将YOLO编码的输出（很多锚框）转换为预测框以及它们的分数，框坐标和类。

    参数：
        yolo_outputs - 编码模型的输出（对于维度为（608,608,3）的图片），包含4个tensors类型的变量：
                        box_confidence ： tensor类型，维度为(None, 19, 19, 5, 1)
                        box_xy         ： tensor类型，维度为(None, 19, 19, 5, 2)
                        box_wh         ： tensor类型，维度为(None, 19, 19, 5, 2)
                        box_class_probs： tensor类型，维度为(None, 19, 19, 5, 80)
        image_shape - tensor类型，维度为（2,），包含了输入的图像的维度，这里是(608.,608.)
        max_boxes - 整数，预测的锚框数量的最大值
        score_threshold - 实数，可能性阈值。
        iou_threshold - 实数，交并比阈值。

    返回：
        scores - tensor类型，维度为(,None)，每个锚框的预测的可能值
        boxes - tensor类型，维度为(4,None)，预测的锚框的坐标
        classes - tensor类型，维度为(,None)，每个锚框的预测的分类
    """

    #将yolo网络的输出结果获取
    box_confidence,box_xy,box_wh,box_class_probs = yolo_outputs

    #将中心格式转换为边界格式
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    #去除低概率值
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)



    #非极大值抑制
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5)

    # 因为输入是(720.,1280.)，而编码模型的输出为（608,608），所以需要对box进行按比例调整,
    # 我的理解是用(720.,1280.)resize到（608,608），但因为框无法进行resize，所以我们通过这个方式对框进行调整
    boxes = yolo_utils.scale_boxes(boxes, image_shape)

    return scores, boxes, classes
def predict(sess, image_file, is_show_info=True, is_plot=True):
    """
    运行存储在sess的计算图以预测image_file的边界框，打印出预测的图与信息。

    参数：
        sess - 包含了YOLO计算图的TensorFlow/Keras的会话。
        image_file - 存储在images文件夹下的图片名称
    返回：
        out_scores - tensor类型，维度为(None,)，锚框的预测的可能值。
        out_boxes - tensor类型，维度为(None,4)，包含了锚框位置信息。
        out_classes - tensor类型，维度为(None,)，锚框的预测的分类索引。
    """
    #图像预处理

    image, image_data = yolo_utils.preprocess_image(r"E:\深度学习\第四课第三周编程作业\Car detection for Autonomous Driving\images\\" + image_file, model_image_size = (608, 608))

    #运行会话并在feed_dict中选择正确的占位符.
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict = {yolo_model.input:image_data, K.learning_phase(): 0})

    #打印预测信息
    if is_show_info:
        print("在" + str(image_file) + "中找到了" + str(len(out_boxes)) + "个锚框。")

    #指定要绘制的边界框的颜色
    colors = yolo_utils.generate_colors(class_names)

    #在图中绘制边界框
    yolo_utils.draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    #保存已经绘制了边界框的图
    image.save(os.path.join("out", image_file), quality=100)

    #打印出已经绘制了边界框的图
    if is_plot:
        output_image = scipy.misc.imread(os.path.join("out", image_file))
        plt.imshow(output_image)

    return out_scores, out_boxes, out_classes

if __name__ == '__main__':
    with tf.Session() as sess:
        class_names = yolo_utils.read_classes(r"E:\深度学习\第四课第三周编程作业\Car detection for Autonomous Driving\model_data/coco_classes.txt")
        anchors = yolo_utils.read_anchors(r"E:\深度学习\第四课第三周编程作业\Car detection for Autonomous Driving\model_data/yolo_anchors.txt")
        image_shape = (720., 1280.)

        yolo_model = load_model(r"E:\深度学习\第四课第三周编程作业\Car detection for Autonomous Driving\model_data/yolo.h5")
        yolo_model.summary()
        yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
        scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
        out_scores, out_boxes, out_classes = predict(sess, r"0083.jpg")

