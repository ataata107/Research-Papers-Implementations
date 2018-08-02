import numpy as np
import os
import time
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.preprocessing import image
from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras.applications.imagenet_utils import preprocess_input
from vgg16 import VGG16
#encoder

image_input = Input(shape=(None, None, 3))
model = VGG16(input_tensor=image_input, include_top=False,weights='imagenet')
model.summary()
last_layer = model.get_layer('max_pooling_with_argmax2d_5').output

#decoder
