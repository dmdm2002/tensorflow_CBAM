import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer, Activation, Concatenate, Conv2D, Add, Dense, MaxPool2D, AvgPool2D, Flatten, multiply
from tensorflow.keras import backend as K


class CAM(object):
    def __init__(self):
        super(CAM, self).__init__()
        self.reduction_ratio = 16

    def build(self, inputs):
        input_shape = inputs.get_shape().as_list()
        # print(input_shape)
        _, h, w, filters = input_shape

        max_pool = MaxPool2D()(inputs)
        max_pool = Flatten()(max_pool)
        max_pool = Dense(filters // self.reduction_ratio)(max_pool)
        max_pool = Activation('relu')(max_pool)
        max_pool = Dense(filters)(max_pool)

        avg_pool = AvgPool2D()(inputs)
        avg_pool = Flatten()(avg_pool)
        avg_pool = Dense(filters // self.reduction_ratio)(avg_pool)
        avg_pool = Activation('relu')(avg_pool)
        avg_pool = Dense(filters)(avg_pool)

        out = Add()([max_pool, avg_pool])
        out = Activation('sigmoid')(out)

        return multiply([inputs, out])

    def __call__(self, inputs):
        return self.build(inputs)


class SAM(object):
    def __init__(self):
        super(SAM, self).__init__()

    def build(self, inputs):
        input_shape = inputs.get_shape().as_list()
        _, h, w, filters = input_shape

        x_max = MaxPool2D(pool_size=(1, 1))(inputs)
        x_avg = AvgPool2D(pool_size=(1, 1))(inputs)
        x_max_avg = Concatenate(axis=3)([x_max, x_avg])

        out = Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid',
                        kernel_initializer='he_normal', use_bias=False)(x_max_avg)

        return multiply([inputs, out])

    def __call__(self, inputs):
        return self.build(inputs)


class CBAM(object):
    def __init__(self):
        super(CBAM, self).__init__()

    def build(self, inputs):
        cam = CAM()
        sam = SAM()

        x = cam(inputs=inputs)
        x = sam(inputs=x)

        return x

    def __call__(self, inputs):
        return self.build(inputs)