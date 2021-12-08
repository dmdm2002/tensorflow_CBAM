import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Conv2D, MaxPool2D, Add, Dense, GlobalAvgPool2D, Activation, BatchNormalization, Input
from keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from Attention import CBAM

class ResNet(object):
    def __init__(self):
        self.cbam = CBAM()

    def resnet_layer(self, inputs, num_filters=16, kernel_size=3, stride=1, activation='relu', batchNorm=True):
        conv = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=stride, padding='same',
                      kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

        x = inputs
        x = conv(x)
        if batchNorm:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)

        return x

    def build(self, input_shape, depth, num_class, attention_module=None):
        block_range = []
        num_filters = 64

        if depth == 50:
            block_range = [3, 4, 6, 3]
        elif depth == 101:
            block_range = [3, 4, 23, 3]

        inputs = Input(shape=input_shape)
        x = self.resnet_layer(inputs=inputs, num_filters=64, kernel_size=7, stride=2)
        x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

        for stage in range(len(block_range)):
            num_filters_out = num_filters * 4

            for stack in range(block_range[stage]):

                if stage != 0 and stack == 0:
                    stride = 2
                else:
                    stride = 1

                unit = self.resnet_layer(inputs=x, kernel_size=1, num_filters=num_filters, stride=stride)
                unit = self.resnet_layer(inputs=unit, kernel_size=3, num_filters=num_filters)
                unit = self.resnet_layer(inputs=unit, kernel_size=1, num_filters=num_filters_out, batchNorm=False, activation=None)
                cbam_feature = self.cbam(unit)

                if stack == 0:
                    x = self.resnet_layer(inputs=x, kernel_size=1, num_filters=num_filters_out, activation=None, stride=stride)

                x = Add()([x, cbam_feature])
                x = Activation('relu')(x)

            # cbam_feature = self.cbam(x)
            # x = Add()([x, cbam_feature])
            num_filters = num_filters * 2

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAvgPool2D()(x)
        output = Dense(num_class, activation='softmax', kernel_initializer='he_normal')(x)

        return keras.Model(inputs=inputs, outputs=output)

    def __call__(self, input_shape, depth, num_class, attention_module=None):
        return self.build(input_shape, depth, num_class, attention_module=None)

# model = ResNet()
# model_res = model.build((224, 224, 3), 50, 2)
# 
# print(model_res.summary())
# plot_model(model_res, to_file='my_res.png', show_shapes=True)