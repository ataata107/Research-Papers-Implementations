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
#encoder decoder
epoch_steps = int(200/1)
val_steps = int(50/1)
image_input = Input(shape=(480, 360, 3))
model = VGG16(input_tensor=image_input, include_top=False,input_shape=(480,360,3),weights='imagenet')
model.summary()

train_list = pd.read_csv(args.train_list,header=None)
val_list = pd.read_csv(args.val_list,header=None)

trainimg_dir = './training_dataset'
trainmsk_dir = './training_mask'
valimg_dir = './validating_dataset'
valmsk_dir = './validating_mask'

train_gen = data_gen_small(trainimg_dir, trainmsk_dir, train_list, batch_size=1, [480, 360], n_labels =2)
val_gen = data_gen_small(valimg_dir, valmsk_dir, val_list, batch_size=1, [480, 360], n_labels =2)
stochastic = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
callbacks = [ModelCheckpoint('bestmodel.h5', monitor='val_loss', save_best_only=True)]

model.compile(loss="categorical_crossentropy", optimizer=stochastic, metrics=["accuracy"])
segnet.fit_generator(train_gen, steps_per_epoch=epoch_steps, epochs=200, validation_data=val_gen, validation_steps=val_steps,callbacks=callbacks)

