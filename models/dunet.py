import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import (BatchNormalization, Conv3D, Conv3DTranspose, Dropout,
                          Input, Layer, ReLU, Reshape, Softmax, concatenate,
                          merge)
from keras.losses import categorical_crossentropy

from .ac_layer import pyramidACBlock

def cross_entropy(y_true, y_pred):
    ce_loss = categorical_crossentropy(y_true, y_pred)
    return ce_loss

class DUNetMixACB(object):
    def __init__(self, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.bn_size = 4
        self.dense_conv = 3
        self.growth_rate = 16
        self.dropout_rate = 0.1
        self.compress_rate = 0.5
        self.num_init_features = 32
        self.conv_size = 3
        self.weight_decay = 0.0001
        self.images = Input(shape=[32,32,32,2], name='input')
        self.num_classes = 4

    def switch_to_deploy(self):
        self.deploy = True

    def encoder(self, input_img):
        img = Conv3D(filters=self.num_init_features, kernel_size=3, strides=1, padding='same',
                    kernel_initializer=keras.initializers.he_normal(),
                    kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name='feature_conv')(input_img)
        img = BatchNormalization(name='feature_bn')(img)
        img = ReLU(name='feature_relu')(img)

        # encode path
        dense1 = self.dense_block(img, 'dense1') # 80 32*32*32
        trans1 = self.transition_down(dense1, 'trans1') # 40

        dense2 = self.dense_block(trans1, 'dense2') # 88 16*16*16
        trans2 = self.transition_down(dense2, 'trans2') # 44

        dense3 = self.dense_block(trans2, 'dense3') # 92 8*8*8

        trans3 = self.transition_down(dense3, 'trans3') # 46
        dense4 = self.dense_block(trans3, 'dense4')  # 94 4*4*4

        return dense1, dense2, dense3, dense4

    def decoder(self, dense1, dense2, dense3, dense4):
        trans4 = self.transition_up(dense4, 'trans4') ## 47

        # decode path
        concat1 = self.SkipConn(dense3, trans4, 'concat1')
        dense5 = self.dense_block(concat1, 'dense5') # 187
        trans5 = self.transition_up(dense5, 'trans5') # 94

        concat2 = self.SkipConn(dense2, trans5, 'concat2')
        dense6 = self.dense_block(concat2, 'dense6') # 229
        trans6 = self.transition_up(dense6, 'trans6') # 115

        concat3 = self.SkipConn(dense1, trans6, 'concat3')
        dense7 = self.dense_block(concat3, 'dense7')  # 242
        
        return dense7

    def build_model(self):
        enc = self.encoder(self.images)
        dec = self.decoder(*enc)

        seg = Conv3D(filters=self.num_classes, kernel_size=1, strides=1,
                     padding='same', dilation_rate=1, use_bias=False,
                     name='seg_conv')(dec)
        seg = Softmax(name='output')(seg)

        model = keras.Model(inputs=self.images, outputs=seg, name='deploy' if self.deploy else 'enrich')
        return model

    def bottle_layer(self, x_in, out_channel, padding='same', use_bias=True, name='bottle'):
        x = Conv3D(filters=out_channel, kernel_size=1, strides=1,
            padding=padding, use_bias=use_bias,
            kernel_initializer=keras.initializers.he_normal(),
            kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name=name+'.conv0')(x_in)
        x = BatchNormalization(name=name+'.bn0')(x)
        x = ReLU(name=name+'.relu0')(x)
        return x

    def dense_layer(self, f_in, name):
        x = self.bottle_layer(f_in, self.bn_size*self.growth_rate, name=name+'.bottle')
        x = self._aconv(x, self.growth_rate, 1, name=name+'.aconv')
        x = Dropout(rate=self.dropout_rate, name=name+'.drop')(x)
        x = concatenate([f_in, x], name=name+'.cat')
        return x

    def dense_block(self, f_in, name='dense_block0'):
        x = f_in
        for i in range(self.dense_conv):
            x = self.dense_layer(x, name=name+'.denselayer{}'.format(i))
        return x
    
    def _aconv(self, x_in, out_channel, stride=1, name=None):
        x = pyramidACBlock(x_in, name+'.pyacb', out_channels=out_channel, strides=stride,
                kernel_initializer=keras.initializers.he_normal(),
                kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                deploy=self.deploy)
        x = ReLU(name=name+'.relu')(x)
        return x

    def _deconv(self, x_in, out_channel, padding='same', name=None):
        x = Conv3DTranspose(filters=out_channel, kernel_size=self.conv_size, strides=2, padding=padding,
                    kernel_initializer=keras.initializers.he_normal(),
                    kernel_regularizer=keras.regularizers.l2(l=self.weight_decay), name=name+'.convtrans')(x_in)
        x = BatchNormalization(name=name+'.bn')(x)
        x = ReLU(name=name+'.relu')(x)
        return x

    def transition_down(self, f_in, name='trans_down'):
        channels = f_in.get_shape()[-1].value
        x = self._aconv(f_in, int(channels * self.compress_rate), 1, name=name+'.conv0')
        x = self._aconv(x, x.get_shape()[-1].value, 2, name=name+'.conv1')
        return x
    
    def transition_up(self, f_in, name='trans_up'):
        channels = f_in.get_shape()[-1].value
        x = self._deconv(f_in, int(channels * self.compress_rate), name=name+'.deconv')
        return x

    def SkipConn(self, enc_f, dec_f, name='skip'):
        """
        f_in_1: the feature map from the encoder path
        f_in_2: the feature map from the decoder path
        """
        return concatenate([enc_f, dec_f], name=name)

