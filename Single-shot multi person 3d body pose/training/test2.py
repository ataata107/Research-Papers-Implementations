import numpy as np
import os
import time
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten

#from imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

model = VGG19(weights='imagenet',include_top=False)
for layer in model.layers:
    print(layer.name)
