

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers



def shortcut(input_tensor, residual_tensor):
    input_shape = K.int_shape(input_tensor)
    residual_shape = K.int_shape(residual_tensor)
    stride_width = int(round(input_shape[1]/residual_shape[1]))
    stride_height = int(round(input_shape[2]/residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]
    
    _shortcut = input_tensor
    #
    if stride_width > 1 or stride_height > 1 or not equal_channels:

        _shortcut = layers.Conv2D(filters=residual_shape[3], 
                                  kernel_size=(1,1),
                                  strides=(stride_width,stride_height),
                                  padding='valid',
                                  kernel_initializer='he_normal')(input_tensor)
        
    return layers.add([_shortcut, residual_tensor])



def residual_block(x, n_filters_out, version=1, 
                   use_bn=True, kernel_size=(3,3)):
    
    if version==1: # resnet v1
        _input_tensor = x
        
        y = layers.Conv2D(n_filters_out, kernel_size=kernel_size, \
            strides=(1,1), padding='same')(x)
        if use_bn:
            y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        
        y = layers.Conv2D(n_filters_out, kernel_size=kernel_size, \
            strides=(1,1), padding='same')(y)
        if use_bn:
            y = layers.BatchNormalization()(y) 

        y = shortcut(_input_tensor, y)
        y = layers.LeakyReLU()(y)

    elif version==2: # resnet v2
        _input_tensor = x

        y = layers.BatchNormalization()(x) 
        y = layers.LeakyReLU()(y)
        y = layers.Conv2D(n_filters_out, kernel_size=kernel_size, \
            strides=(1,1), padding='same')(y)

        y = layers.BatchNormalization()(y) 
        y = layers.LeakyReLU()(y) 
        y = layers.Conv2D(n_filters_out, kernel_size=kernel_size, \
            strides=(1,1), padding='same')(y)

        y = shortcut(_input_tensor, y)
    return y 