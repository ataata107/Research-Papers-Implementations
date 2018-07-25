import tensorflow as tf
import numpy as np
import time
from model import ResNet50
def train():
    Y_hat, model_params = ResNet50()
    #for k, v in model_params.items():
        #print(k, v)
        #time.sleep(2)
train()
