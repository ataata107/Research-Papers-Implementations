import math
import os
import re
import sys
import pandas
from functools import partial

import keras.backend as K
from keras.applications.vgg19 import VGG19
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.convolutional import Conv2D