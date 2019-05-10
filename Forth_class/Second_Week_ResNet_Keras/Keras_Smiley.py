import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import kt_utils

import keras.backend as K

K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def build_model(input_shape):
    input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(input)
    X = Conv2D(32, kernel_size=(7,7), strides=(1,1), padding='same', name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D(pool_size=(2, 2), name='pool0')(X)
    X = Flatten()(X)
    output = Dense(1, activation='sigmoid', name='fc0')(X)

    model = Model(inputs=input,outputs=output)

    return model
if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    model = build_model(X_train.shape[1:])

    model.summary()

    model.compile("adam",loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(x=X_train,y=Y_train,batch_size=64,epochs=50)

    preds = model.evaluate(x=X_test,y= Y_test,batch_size=64,verbose=1)

    print("误差值 = " + str(preds[0]))
    print("准确度 = " + str(preds[1]))