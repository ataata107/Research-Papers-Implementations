import math
import os
import re
import sys
import pandas
from functools import partial

import keras.backend as K
from keras.applications.resnet50 import ResNet50
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.convolutional import Conv2D

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.cmu_model import get_training_model
from training.optimizers import MultiSGD
from training.dataset import get_dataflow, batch_dataflow

batch_size = 8
base_lr = 4e-5 # 2e-5
momentum = 0.9
weight_decay = 5e-4
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 200000 # 600000

weights_best_file = "weights.best.h5"
training_log = "training.csv"
logs_dir = "./logs"

def get_last_epoch():
    """
    Retrieves last epoch from log file updated during training.
    :return: epoch number
    """
    data = pandas.read_csv(training_log)
    return max(data['epoch'].values)

def restore_weights(weights_best_file, model):
    """
    Restores weights from the checkpoint file if exists or
    preloads the first layers with VGG19 weights
    :param weights_best_file:
    :return: epoch number to use to continue training. last epoch + 1 or 0
    """
    # load previous weights or vgg19 if this is the first run
    if os.path.exists(weights_best_file):
        print("Loading the best weights...")

        model.load_weights(weights_best_file)

        return get_last_epoch() + 1
    else:
        print("Loading vgg19 weights...")

        model = ResNet50(weights='imagenet',include_top=False,input_shape=(484,484,3))

        for layer in model.layers:
            if layer.name in from_vgg:
                vgg_layer_name = from_vgg[layer.name]
                layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
                print("Loaded VGG19 layer: " + vgg_layer_name)

        return 0


if __name__ == '__main__':
    # get the model
    model = get_training_model(weight_decay)
