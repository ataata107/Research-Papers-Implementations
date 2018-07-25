# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten,Concatenate
from keras.layers import Conv2D,Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
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

def id2(input_tensor, kernel_size, filters, stage, block,weight_decay,strides):
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

    x = Conv2DTranspose(filters1, (4, 4),strides=strides,padding='same', name=conv_name_base + '2a',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
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
    #x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    #x = layers.add([x, input_tensor])
    #x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),weight_decay):
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



def conv_block1(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),weight_decay):
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

    x = Conv2DTranspose(filters2, kernel_size, padding='same', strides=strides,
               name=conv_name_base + '2b',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1,1), name=conv_name_base + '2c',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2DTranspose(filters3, kernel_size, strides=strides,padding='same',
                      name=conv_name_base + '1',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x




def get_training_model(weight_decay)
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
    np_branch12 = 38
    np_branch2=34+17*3
    img_input_shape = (484, 484, 3)
    vec_input_shape_br1=(int(484/8),int(484/8),34)
    heat_input_shape_br1=(int(484/8),int(484/8),17)
    vec_input_shape_br2=(int(484/4),int(484/4),17*3)
    heat_input_shape_br2=(int(484/4),int(484/4),17)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input_br1 = Input(shape=vec_input_shape_br1)
    heat_weight_input_br1 = Input(shape=heat_input_shape_br1)
    vec_weight_input_br2 = Input(shape=vec_input_shape_br2)
    heat_weight_input_br2 = Input(shape=heat_input_shape_br2)
    
    inputs.append(img_input)
    inputs.append(vec_weight_input_br1)
    inputs.append(heat_weight_input_br1)
    inputs.append(vec_weight_input_br2)
    inputs.append(heat_weight_input_br2)

    img_normalized = Lambda(lambda x:x /256 - 0.5)(img_input)
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_normalized)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1',kernel_regularizer=kernel_reg,bias_regularizer=bias_reg,kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),(weight_decay,0))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',(weight_decay,0))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',(weight_decay,0))

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',(weight_decay,0))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',(weight_decay,0))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',(weight_decay,0))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',(weight_decay,0))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',(weight_decay,0))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',(weight_decay,0))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',(weight_decay,0))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d',(weight_decay,0))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e',(weight_decay,0))
    x1 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f',(weight_decay,0))

    x = conv_block(x1, 3, [512, 512, 1024], stage=5, block='a',strides=(1,1),(weight_decay,0))
    x2 = id1(x, 3, [256, 256, 256], stage=5, block='b',(weight_decay,0),strides=(1,1))
    x = id2(x2, 3, [128, 128, 57], stage=5, block='c',(weight_decay,0),strides=(2,2))
    #Slice1
    heat_1 = K.slice(x, [0,0,0], [62,62,18])
    w1 = apply_mask(heat_1, vec_weight_input, heat_weight_input, np_branch11, 1, 1)
    PAF_1 = K.slice(x,[0,0,19],[62,62,56])
    w2 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch12, 1, 1)
    outputs.append(w1)
    outputs.append(w2)
    #Slice1
    y=Concatenate(axis=-1)([x1,x2])

    y = conv_block(y, 3, [512, 512, 1024], stage=6, block='a', strides=(1, 1),(weight_decay,0))
    y = identity_block(y, 3, [512, 512, 1024], stage=6, block='b',(weight_decay,0))
    y = identity_block(y, 3, [512, 512, 1024], stage=6, block='c',(weight_decay,0))

    y = conv_block1(y, 4, [512, 512, 1024], stage=7, block='a',(weight_decay,0), strides=(2, 2))
    y = identity_block(y, 3, [512, 512, 1024], stage=7, block='b',(weight_decay,0))
    y = identity_block(y, 3, [512, 512, 1024], stage=7, block='c',(weight_decay,0))

    y=Concatenate(axis=-1)([x,y])
    
    y = conv_block(y, 3, [512, 512, 1024], stage=8, block='a', strides=(1, 1),(weight_decay,0))
    y = id1(y, 3, [256,256,256], stage=8, block='b',(weight_decay,0),strides=(1, 1))
    y = id2(y, 5, [128,128,84], stage=8, block='c',(weight_decay,0),strides=(2, 2))

    #Slice2
    heat_1 = K.slice(y, [0,0,0], [62,62,20])
    heat_2  = K.slice(y, [0,0,21], [62,62,41])
    heat_3 = K.slice(x, [0,0,42], [62,62,62])
    heat_4 = K.slice(x,[0,0,63],[62,62,83])
    #Slice2

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


if __name__ == '__main__':
    model = ResNet50(include_top=True, weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
