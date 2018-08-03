import numpy as np
import os
import time
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras.applications.imagenet_utils import preprocess_input
from model import VGG16

model = load_model('bestmodel.h5')
original_img = cv2.imread(img_name+".jpg")[:, :, ::-1]
resized_img = cv2.resize(original_img, [480,360]+[3])
array_img = img_to_array(resized_img)/255
array_2d = model.predict(array_img)
mask = catelab_inverse(array_2d)
cv2.imwrite('mask.jpg',mask)

def catelab_invese(arrr):
    counter=0
    x=np.zeros([480,360,1])
    for i in range(480):
        for j in range(360):
            if(arrr[i+j,0]>arrr[i+j,1]):
                x[i,j]=255

    return x
