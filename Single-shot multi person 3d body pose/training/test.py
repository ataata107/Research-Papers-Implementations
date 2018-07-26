import numpy as np
import os
import time
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten

#from imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from_resnet = {
    'conv1_1': 'conv1',
    'conv1_2': 'res2a_branch2a',
    'conv2_1': 'res2a_branch2b',
    'conv2_2': 'res2a_branch2c',
    'conv3_1': 'res2a_branch1',
    'conv3_2': 'res2b_branch2a',
    'conv3_3': 'res2b_branch2b',
    'conv3_4': 'res2b_branch2c',
    'conv4_1': 'res2c_branch2a',
    'conv4_2': 'res2c_branch2b',
    'conv5': 'res2c_branch2c',
    'conv6': 'res3a_branch2a',
    'conv7': 'res3a_branch2b',
    'conv8': 'res3a_branch2c',
    'conv9': 'res3a_branch1',
    'conv10': 'res3b_branch2a',
    'conv11': 'res3b_branch2b',
    'conv12': 'res3b_branch2c',
    'conv13': 'res3c_branch2a',
    'conv14': 'res3c_branch2b',
    'conv15': 'res3c_branch2c',
    'conv16': 'res3d_branch2a',
    'conv17': 'res3d_branch2b',
    'conv18': 'res3d_branch2c',
    'conv19': 'res4a_branch2a',
    'conv20': 'res4a_branch2b',
    'conv21': 'res4a_branch2c',
    'conv22': 'res4a_branch1',
    'conv23': 'res4b_branch2a',
    'conv24': 'res4b_branch2b',
    'conv25': 'res4b_branch2c',
    'conv26': 'res4c_branch2a',
    'conv27': 'res4c_branch2b',
    'conv28': 'res4c_branch2c',
    'conv29': 'res4d_branch2a',
    'conv30': 'res4d_branch2b',
    'conv31': 'res4d_branch2c',
    'conv32': 'res4e_branch2a',
    'conv33': 'res4e_branch2b',
    'conv34': 'res4e_branch2c',
    'conv35': 'res4f_branch2a',
    'conv36': 'res4f_branch2b',
    'conv37': 'res4f_branch2c'
    
    
}
for i,j in from_resnet.items():
    print(j)
resnet_model = ResNet50(weights='imagenet',include_top=False)
for layer in resnet_model.layers:
    if (layer.name in from_resnet.values()):
        print("true")
        
