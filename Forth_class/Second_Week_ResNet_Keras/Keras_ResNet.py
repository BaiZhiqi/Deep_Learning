import numpy as np
import tensorflow as tf

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform

import pydot
from IPython.display import SVG
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import resnets_utils


def identity_block(X, f, filters, stage, block):
    '''
    :param X:输入
    :param f: 整数，指定主路径中间的CONV窗口的维度
    :param filters:卷积核个数
    :param stage:层数
    :param block:
    '''

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    f1, f2, f3 = filters

    short_cut = X

    # 统一初始化器
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(name=bn_name_base + '2a')(X)

    X = Activation('relu')(X)

    X = Conv2D(filters=f2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(name=bn_name_base + '2b')(X)

    X = Activation('relu')(X)

    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(name=bn_name_base + '2c')(X)

    X = Add()([X, short_cut])

    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    '''
        :param X:输入
        :param f: 整数，指定主路径中间的CONV窗口的维度
        :param filters:卷积核个数
        :param stage:层数
        :param block:
        '''

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    f1, f2, f3 = filters

    short_cut = X

    # 统一初始化器
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(name=bn_name_base + '2a')(X)

    X = Activation('relu')(X)

    X = Conv2D(filters=f2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(name=bn_name_base + '2b')(X)

    X = Activation('relu')(X)

    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(name=bn_name_base + '2c')(X)

    # 这里s是为了和主路径尺寸保持一致
    X1 = Conv2D(filters=f3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                kernel_initializer=glorot_uniform(seed=0))(short_cut)

    X1 = BatchNormalization(name=bn_name_base + '1')(X1)

    X = Add()([X, X1])

    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
    input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(input)
    # stage1
    X = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), name='conv1')(X)
    X = BatchNormalization(name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # stage2

    X = convolutional_block(X, 3, [64, 64, 256], 2, 'a', s=1)
    X = identity_block(X, 3, [64, 64, 256], 2, 'b')
    X = identity_block(X, 3, [64, 64, 256], 2, 'c')

    # stage3
    X = convolutional_block(X, 3, [128, 128, 512], 3, 'a', s=2)
    X = identity_block(X, 3, [128, 128, 512], 3, 'b')
    X = identity_block(X, 3, [128, 128, 512], 3, 'c')
    X = identity_block(X, 3, [128, 128, 512], 3, 'd')

    # stage4

    X = convolutional_block(X, 3, [256, 256, 1024], 4, 'a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], 4, 'b')
    X = identity_block(X, 3, [256, 256, 1024], 4, 'c')
    X = identity_block(X, 3, [256, 256, 1024], 4, 'd')
    X = identity_block(X, 3, [256, 256, 1024], 4, 'e')
    X = identity_block(X, 3, [256, 256, 1024], 4, 'f')

    # stage5

    X = convolutional_block(X, 3, [512, 512, 2048], 5, 'a', s=2)
    X = identity_block(X, 3, [256, 256, 2048], 5, 'b')
    X = identity_block(X, 3, [256, 256, 2048], 5, 'c')

    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    X = Flatten()(X)
    output = Dense(classes, activation='softmax', name="fc" + str(classes))(X)

    model = Model(inputs=input, outputs=output)

    return model


if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = resnets_utils.load_dataset()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = resnets_utils.convert_to_one_hot(Y_train_orig, 6).T
    Y_test = resnets_utils.convert_to_one_hot(Y_test_orig, 6).T

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    model = ResNet50(input_shape=(64, 64, 3), classes=6)

    model.summary()

    plot_model(model, to_file='model.png')
    SVG(model_to_dot(model).create(prog='dot', format='svg'))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, Y_train, epochs=2, batch_size=32)

    preds = model.evaluate(X_test, Y_test)

    print("误差值 = " + str(preds[0]))
    print("准确率 = " + str(preds[1]))
