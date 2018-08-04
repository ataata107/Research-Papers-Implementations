# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input,UpSampling2D
from keras.layers import Conv2D,Activation
from keras.layers import MaxPooling2D,BatchNormalization
from keras.layers import GlobalMaxPooling2D,Reshape
from keras.layers import GlobalAveragePooling2D
from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    pool_size=(2,2)
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3),  padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_bn1')(x)
    x=  Activation("relu")(x)
    x = Conv2D(64, (3, 3),  padding='same', name='block1_conv2')(x)
    x = BatchNormalization(name='block1_bn2')(x)
    x=  Activation("relu")(x)
    x, mask_1 = MaxPoolingWithArgmax2D(pool_size)(x)

    # Block 2
    x = Conv2D(128, (3, 3),  padding='same', name='block2_conv1')(x)
    x = BatchNormalization(name='block2_bn1')(x)
    x=  Activation("relu")(x)
    x = Conv2D(128, (3, 3),  padding='same', name='block2_conv2')(x)
    x = BatchNormalization(name='block2_bn2')(x)
    x=  Activation("relu")(x)
    x, mask_2 = MaxPoolingWithArgmax2D(pool_size)(x)

    # Block 3
    x = Conv2D(256, (3, 3),  padding='same', name='block3_conv1')(x)
    x = BatchNormalization(name='block3_bn1')(x)
    x=  Activation("relu")(x)
    x = Conv2D(256, (3, 3),  padding='same', name='block3_conv2')(x)
    x = BatchNormalization(name='block3_bn2')(x)
    x=  Activation("relu")(x)
    x = Conv2D(256, (3, 3),  padding='same', name='block3_conv3')(x)
    x = BatchNormalization(name='block3_bn3')(x)
    x=  Activation("relu")(x)
    x, mask_3 = MaxPoolingWithArgmax2D(pool_size)(x)

    # Block 4
    x = Conv2D(512, (3, 3),  padding='same', name='block4_conv1')(x)
    x = BatchNormalization(name='block4_bn1')(x)
    x=  Activation("relu")(x)
    x = Conv2D(512, (3, 3),  padding='same', name='block4_conv2')(x)
    x = BatchNormalization(name='block4_bn2')(x)
    x=  Activation("relu")(x)
    x = Conv2D(512, (3, 3),  padding='same', name='block4_conv3')(x)
    x = BatchNormalization(name='block4_bn3')(x)
    x=  Activation("relu")(x)
    x, mask_4 = MaxPoolingWithArgmax2D(pool_size)(x)

    # Block 5
    x = Conv2D(512, (3, 3),  padding='same', name='block5_conv1')(x)
    x = BatchNormalization(name='block5_bn1')(x)
    x=  Activation("relu")(x)
    x = Conv2D(512, (3, 3),  padding='same', name='block5_conv2')(x)
    x = BatchNormalization(name='block5_bn2')(x)
    x=  Activation("relu")(x)
    x = Conv2D(512, (3, 3),  padding='same', name='block5_conv3')(x)
    x = BatchNormalization(name='block5_bn3')(x)
    x=  Activation("relu")(x)
    x, mask_5 = MaxPoolingWithArgmax2D(pool_size)(x)

    #Decoder
    # Block6
    #x = MaxUnpooling2D(pool_size)([x, mask_5])
    x = UpSampling2D(size=(2, 2), data_format=None)(x)
    x = Conv2D(512, (3, 3),  padding='same', name='block6_conv1')(x)
    x = BatchNormalization(name='block6_bn1')(x)
    x=  Activation("relu")(x)
    x = Conv2D(512, (3, 3),  padding='same', name='block6_conv2')(x)
    x = BatchNormalization(name='block6_bn2')(x)
    x=  Activation("relu")(x)
    x = Conv2D(512, (3, 3),  padding='same', name='block6_conv3')(x)
    x = BatchNormalization(name='block6_bn3')(x)
    x=  Activation("relu")(x)

    #Block7
    #x = MaxUnpooling2D(pool_size)([x, mask_4])
    x = UpSampling2D(size=(2, 2), data_format=None)(x)
    x = Conv2D(512, (3, 3),  padding='same', name='block7_conv1')(x)
    x = BatchNormalization(name='block7_bn1')(x)
    x=  Activation("relu")(x)
    x = Conv2D(512, (3, 3),  padding='same', name='block7_conv2')(x)
    x = BatchNormalization(name='block7_bn2')(x)
    x=  Activation("relu")(x)
    x = Conv2D(256, (3, 3),  padding='same', name='block7_conv3')(x)
    x = BatchNormalization(name='block7_bn3')(x)
    x=  Activation("relu")(x)

    #Block8
    #x = MaxUnpooling2D(pool_size)([x, mask_3])
    x = UpSampling2D(size=(2, 2), data_format=None)(x)
    x = Conv2D(256, (3, 3),  padding='same', name='block8_conv1')(x)
    x = BatchNormalization(name='block8_bn1')(x)
    x=  Activation("relu")(x)
    x = Conv2D(256, (3, 3),  padding='same', name='block8_conv2')(x)
    x = BatchNormalization(name='block8_bn2')(x)
    x=  Activation("relu")(x)
    x = Conv2D(128, (3, 3),  padding='same', name='block8_conv3')(x)
    x = BatchNormalization(name='block8_bn3')(x)
    x=  Activation("relu")(x)

    #Block9
    #x = MaxUnpooling2D(pool_size)([x, mask_2])
    x = UpSampling2D(size=(2, 2), data_format=None)(x)
    x = Conv2D(128, (3, 3),  padding='same', name='block9_conv1')(x)
    x = BatchNormalization(name='block9_bn1')(x)
    x=  Activation("relu")(x)
    x = Conv2D(64, (3, 3),  padding='same', name='block9_conv2')(x)
    x = BatchNormalization(name='block9_bn2')(x)
    x=  Activation("relu")(x)

    #Block10
    #x = MaxUnpooling2D(pool_size)([x, mask_1])
    x = UpSampling2D(size=(2, 2), data_format=None)(x)
    x = Conv2D(64, (3, 3),  padding='same', name='block10_conv1')(x)
    x = BatchNormalization(name='block10_bn1')(x)
    x=  Activation("relu")(x)
    x = Conv2D(2, (1, 1),  padding='valid', name='block10_conv2')(x)
    x = BatchNormalization(name='block10_bn2')(x)
    x = Reshape((input_shape[0] * input_shape[1], 2))(x)
    x = Activation('softmax')(x)
    print(x.shape)
    

    

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')
    '''
    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
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
        '''
    return model


if __name__ == '__main__':
    model = VGG16(include_top=True, weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
