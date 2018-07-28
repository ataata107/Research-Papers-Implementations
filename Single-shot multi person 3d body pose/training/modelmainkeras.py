# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np
import warnings
#import tensorflow as tf
from convo import Conv2DTranspose
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten,Concatenate
from keras.layers import Conv2D, Lambda,Deconvolution2D#,Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.initializers import random_normal,constant
from keras.preprocessing import image
from keras.layers.merge import Multiply
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

assert K.image_data_format() == 'channels_last'
#WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
#WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)
def apply_mask(x, mask1, mask2, num_p, stage, branch):
    
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if num_p == 38:
        w = Multiply(name=w_name)([x, mask1]) # vec_weight

    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w


def identity_block(input_tensor, kernel_size, filters, stage, block,weight_decay):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None
    
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def id1(input_tensor, kernel_size, filters, stage, block,weight_decay,strides):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None
    
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),strides=strides, name=conv_name_base + '2a',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, kernel_size,padding='same', name=conv_name_base + '2c',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    #x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def id2(input_tensor, kernel_size, filters, stage, block,weight_decay,strides,output_shape = None):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    print(input_tensor.shape)
    #print(output_shape)
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None
    
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    #x = Deconvolution2D(filters1, 4, 4, output_shape=(None, 46, 46, 128),border_mode='same')(input_tensor)

    x = Conv2DTranspose(filters1,(4, 4),strides=strides,padding='same', output_shape=output_shape,name=conv_name_base + '2a',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,
                        kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(input_tensor)
    #x = K.tensorflow_backend.conv2d_transpose(input_tensor, (4,4),output_shape=(None,None,128), strides=strides,padding='same', data_format=None)
    #print(x.shape)
    #x=input_tensor
    #x = UpSampling2D(size=(2, 2), data_format=None)(x)
    #print(x.shape)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, kernel_size,padding='same', name=conv_name_base + '2c',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    #x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    #x = layers.add([x, input_tensor])
    #x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),weight_decay=(0,0)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None
    
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'


    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x



def conv_block1(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),weight_decay=(0,0),output_shape=None):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None
    
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
               name=conv_name_base + '2a',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(filters2, kernel_size, padding='same', strides=strides,output_shape=output_shape,
               name=conv_name_base + '2b',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    #print(x.shape)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1,1), name=conv_name_base + '2c',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2DTranspose(filters3, kernel_size, strides=strides,padding='same',output_shape=(46,46,1024),
                      name=conv_name_base + '1',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(input_tensor)
    #print(shortcut.shape)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x




def get_training_model(weight_decay):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """


    np_branch11= 19
    #print("done")
    np_branch12 = 38
    np_branch2=34+17*3
    img_input_shape = (368, 368, 3)
    vec_input_shape_br1=(None,None,38)
    heat_input_shape_br1=(None,None,19)
    vec_input_shape_br2=(None,None,17*3)
    heat_input_shape_br2=(None,None,17)

    inputs1 = []
    inputs2=[]
    outputs_br1 = []
    outputs_br2=[]

    img_input = Input(shape=img_input_shape)
    vec_weight_input_br1 = Input(shape=vec_input_shape_br1)
    heat_weight_input_br1 = Input(shape=heat_input_shape_br1)
    vec_weight_input_br2 = Input(shape=vec_input_shape_br2)
    heat_weight_input_br2 = Input(shape=heat_input_shape_br2)
    
    inputs1.append(img_input)
    inputs1.append(vec_weight_input_br1)
    inputs1.append(heat_weight_input_br1)
    inputs2.append(img_input)
    inputs2.append(vec_weight_input_br2)
    inputs2.append(heat_weight_input_br2)

    img_normalized = Lambda(lambda x:x /256 - 0.5)(img_input)
    #print(img_normalized.shape)
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    kernel_reg = l2(0) 
    bias_reg = l2(0) 

    x = ZeroPadding2D((3, 3))(img_normalized)
    
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),weight_decay = (weight_decay,0))
    
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',weight_decay = (weight_decay,0))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',weight_decay = (weight_decay,0))

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',weight_decay = (weight_decay,0))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',weight_decay = (weight_decay,0))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',weight_decay = (weight_decay,0))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',weight_decay = (weight_decay,0))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',weight_decay = (weight_decay,0))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',weight_decay = (weight_decay,0))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',weight_decay = (weight_decay,0))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d',weight_decay = (weight_decay,0))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e',weight_decay = (weight_decay,0))
    
    x1 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f',weight_decay = (weight_decay,0))

    x = conv_block(x1, 3, [512, 512, 1024], stage=5, block='a',strides=(1,1),weight_decay = (weight_decay,0))
    x2 = id1(x, 3, [256, 256, 256], stage=5, block='b',weight_decay = (weight_decay,0),strides=(1,1))
    #print(x2.shape)
    x3 = id2(x2, 3, [128, 128, 38], stage=5, block='c',weight_decay = (weight_decay,0),strides=(2,2),output_shape = (46,46,128))
    x4 = id2(x2, 3, [128, 128, 19], stage=6, block='c',weight_decay = (weight_decay,0),strides=(2,2),output_shape = (46,46,128))
    print(x3.shape)
    print(x4.shape)
    #
    #Slice1
    #heat_1 = Lambda(lambda x: x[:,:,:,:19], output_shape=(None,None,None,19),name='bhola')(x)
    
    #print(heat_1.shape)
    #heat_1 = tf.convert_to_tensor(heat_1)
    #print(heat_1.shape)
    #print(PAF_1.shape)
    #print(heat_weight_input_br1.shape)
    w1 = apply_mask(x4, vec_weight_input_br1, heat_weight_input_br1, np_branch11, 1, 2)
    #print(w1.shape)
    PAF_1 = Lambda(lambda x: x[:,:,:,19:], output_shape=(None,None,None,38),name='hola')(x)
    w2 = apply_mask(x3, vec_weight_input_br1, heat_weight_input_br1, np_branch12, 1, 1)
    #print(w2.shape)
    
    outputs_br1.append(w2)
    outputs_br1.append(w1)
    #outputs_br1.append(w2)
    #outputs_br1.append(w1)
    #outputs_br1.append(w2)
    #outputs_br1.append(w1)
    #outputs_br1.append(w2)
    #outputs_br1.append(w1)
    #outputs_br1.append(w2)
    #outputs_br1.append(w1)
    #outputs_br1.append(w2)
    #outputs_br1.append(w1)

    #Slice1
    #y=Concatenate(axis=-1)([x1,x2])

    #y = conv_block(y, 3, [512, 512, 1024], stage=6, block='a', strides=(1, 1),weight_decay = (weight_decay,0))
    #y = identity_block(y, 3, [512, 512, 1024], stage=6, block='b',weight_decay = (weight_decay,0))
    #y = identity_block(y, 3, [512, 512, 1024], stage=6, block='c',weight_decay = (weight_decay,0))

    #y = conv_block1(y, 4, [512, 512, 1024], stage=7, block='a',weight_decay = (weight_decay,0), strides=(2, 2),output_shape=(46,46,512))
    #y = identity_block(y, 3, [512, 512, 1024], stage=7, block='b',weight_decay = (weight_decay,0))
    #y = identity_block(y, 3, [512, 512, 1024], stage=7, block='c',weight_decay = (weight_decay,0))

    #y=Concatenate(axis=-1)([x,y])
    
    #y = conv_block(y, 3, [512, 512, 1024], stage=8, block='a', strides=(1, 1),weight_decay = (weight_decay,0))
    #y = id1(y, 3, [256,256,256], stage=8, block='b',weight_decay = (weight_decay,0),strides=(1, 1))
    #y = id2(y, 5, [128,128,84], stage=8, block='c',weight_decay = (weight_decay,0),strides=(2, 2),output_shape=(92,92,128))

    #Slice2
    #heat_1 = Lambda(lambda x: x[:,:,:,:21], output_shape=(None,None,None,21))(y)
    #orpm_x  = Lambda(lambda x: x[:,:,:,21:42], output_shape=(None,None,None,21))(y)
    #orpm_y = Lambda(lambda x: x[:,:,:,42:63], output_shape=(None,None,None,21))(y)
    #orpm_z = Lambda(lambda x: x[:,:,:,63:84], output_shape=(None,None,None,21))(y)
    #outputs_br2.append(heat_1)
    #outputs_br2.append(orpm_x)
    #outputs_br2.append(orpm_y)
    #outputs_br2.append(orpm_z)
    #Slice2

    model1 = Model(inputs=inputs1, outputs=outputs_br1)
    #model2 = Model(inputs=inputs2, outputs=outputs_br2)


    return model1#,model2



model = get_training_model(0)


