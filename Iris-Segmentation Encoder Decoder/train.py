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
#encoder decoder

image_input = Input(shape=(256, 256, 3))
model = VGG16(input_tensor=image_input, include_top=False,input_shape=(256,256,3),weights='imagenet')
model.summary()


