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
import pandas as pd
from generator import data_gen_small
#encoder decoder
epoch_steps = 6
val_steps = 3
image_input = Input(shape=(480, 360, 3))
#print(image_input.shape)
model = VGG16(input_tensor=image_input, include_top=False,input_shape=(480,360,3),weights='imagenet')
#model.summary()

train_list = pd.read_csv('./train.csv',header=None)
val_list = pd.read_csv('./test.csv',header=None)

trainimg_dir = './training_dataset'
trainmsk_dir = './training_mask'
valimg_dir = './validating_dataset'
valmsk_dir = './validating_mask'

x,y = data_gen_small(trainimg_dir, trainmsk_dir, 6, batch_size=1, dims=[480, 360], n_labels =2)
print(x.shape)
x1,y1 = data_gen_small(valimg_dir, valmsk_dir, 3, batch_size=1, dims=[480, 360], n_labels =2)
stochastic = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
callbacks = [ModelCheckpoint('bestmodel.h5', monitor='val_loss', save_best_only=True)]

model.compile(loss="categorical_crossentropy", optimizer=stochastic, metrics=["accuracy"])
#model.fit_generator(train_gen, steps_per_epoch=epoch_steps, epochs=200, validation_data=val_gen, validation_steps=val_steps,callbacks=callbacks)
model.fit(x=x, y=y, batch_size=1, epochs=10, verbose=1, callbacks=callbacks, validation_split=0.0, validation_data=(x1,y1), shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
