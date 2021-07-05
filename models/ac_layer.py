import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
from keras.layers import BatchNormalization, Conv3D, Add, ZeroPadding3D, Lambda, concatenate

def ACBlock(input_tensor, name=None, filters=None, ksize=None, strides=(1, 1),
    momentum=0.99, epsilon=0.001, moving_mean_initializer='zeros', moving_variance_initializer='ones',
    kernel_initializer='glorot_uniform', kernel_regularizer=None, beta_initializer='zeros', gamma_initializer='ones', deploy=False):
    if name == None:
        raise Exception("Please set the name of this convolution layer")
    if filters == None:
        raise Exception('Please set the number of filters this convolution layer')
    if ksize == None:
        raise Exception('Please set the kernel size of this convolution layer')
    if not isinstance(ksize, int):
        raise Exception('kernel size must be an integer')

    pad_size = ksize // 2

    if deploy:
        outs_fusion = ZeroPadding3D(padding=(pad_size, pad_size, pad_size), name=name+'.fused_pad')(input_tensor)
        outs_fusion = Conv3D(filters=filters, kernel_size=(ksize, ksize, ksize), strides=strides, padding='valid', kernel_initializer=kernel_initializer, 
                        kernel_regularizer=kernel_regularizer, use_bias=True, name=name+'.fused_conv')(outs_fusion)
        return outs_fusion
        
    else:
        outs_nxnxn = ZeroPadding3D(padding=(pad_size, pad_size, pad_size), name=name+'.pad_nxnxn')(input_tensor)
        outs_1xnxn = ZeroPadding3D(padding=(0, pad_size, pad_size), name=name+'.pad_1xnxn')(input_tensor)
        outs_nx1xn = ZeroPadding3D(padding=(pad_size, 0, pad_size), name=name+'.pad_nx1xn')(input_tensor)
        outs_nxnx1 = ZeroPadding3D(padding=(pad_size, pad_size, 0), name=name+'_pad_nxnx1')(input_tensor)

        outs_nxnxn = Conv3D(filters=filters, kernel_size=(ksize, ksize, ksize), strides=strides, padding='valid', kernel_initializer=kernel_initializer, 
                        kernel_regularizer=kernel_regularizer, use_bias=False, name=name+'.conv_nxnxn')(outs_nxnxn)
        outs_1xnxn = Conv3D(filters=filters, kernel_size=(1, ksize, ksize), strides=strides, padding='valid', kernel_initializer=kernel_initializer, 
                        kernel_regularizer=kernel_regularizer, use_bias=False, name=name+'.conv_1xnxn')(outs_1xnxn)
        outs_nx1xn = Conv3D(filters=filters, kernel_size=(ksize, 1, ksize), strides=strides, padding='valid', kernel_initializer=kernel_initializer, 
                        kernel_regularizer=kernel_regularizer, use_bias=False, name=name+'.conv_nx1xn')(outs_nx1xn)
        outs_nxnx1 = Conv3D(filters=filters, kernel_size=(ksize, ksize, 1), strides=strides, padding='valid', kernel_initializer=kernel_initializer, 
                        kernel_regularizer=kernel_regularizer, use_bias=False, name=name+'.conv_nxnx1')(outs_nxnx1)

        outs_nxnxn = BatchNormalization(momentum=momentum, epsilon=epsilon, beta_initializer=beta_initializer,
                                        gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
                                        moving_variance_initializer=moving_variance_initializer, name=name+'.bn_nxnxn')(outs_nxnxn)
        outs_1xnxn = BatchNormalization(momentum=momentum, epsilon=epsilon, beta_initializer=beta_initializer,
                                        gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
                                        moving_variance_initializer=moving_variance_initializer, name=name+'.bn_1xnxn')(outs_1xnxn)
        outs_nx1xn = BatchNormalization(momentum=momentum, epsilon=epsilon, beta_initializer=beta_initializer,
                                        gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
                                        moving_variance_initializer=moving_variance_initializer, name=name+'.bn_nx1xn')(outs_nx1xn)
        outs_nxnx1 = BatchNormalization(momentum=momentum, epsilon=epsilon, beta_initializer=beta_initializer,
                                        gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
                                        moving_variance_initializer=moving_variance_initializer, name=name+'.bn_nxnx1')(outs_nxnx1)
        
        added = Add(name = name+'.add')([outs_nxnxn, outs_1xnxn, outs_nx1xn, outs_nxnx1])

        return added

# this is the MixACB function, which is originally named pyramidACB.
def pyramidACBlock(input_tensor, name=None, in_channels_ratio=[3, 1], out_channels_ratio=[3, 1], out_channels=16, ksize=[3, 5], strides=1,
    kernel_initializer='glorot_uniform', kernel_regularizer=None, deploy=False):
    
    in_channels = input_tensor.get_shape()[-1].value
    in_channels_for_conv3 = in_channels // np.sum(in_channels_ratio) * in_channels_ratio[0]
    out_channels_for_conv3 = out_channels // np.sum(out_channels_ratio) * out_channels_ratio[0]
    out_channels_for_conv5 = out_channels - out_channels_for_conv3

    input_tensor_for_conv3 = Lambda(lambda x: x[:,:,:,:,0:in_channels_for_conv3])(input_tensor)
    output_tensor_for_conv3 = ACBlock(input_tensor_for_conv3, name=name+'.pyacb_3', filters=out_channels_for_conv3, ksize=ksize[0], strides=strides,
                        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, deploy=deploy)

    input_tensor_for_conv5 = Lambda(lambda x: x[:,:,:,:,in_channels_for_conv3:])(input_tensor)
    output_tensor_for_conv5 = ACBlock(input_tensor_for_conv5, name=name+'.pyacb_5', filters=out_channels_for_conv5, ksize=ksize[1], strides=strides,
                        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, deploy=deploy)
    outputs = concatenate([output_tensor_for_conv3, output_tensor_for_conv5], axis=-1, name=name+'.concat_3_5')
    return outputs

