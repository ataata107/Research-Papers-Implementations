import math
import random

import cv2
import numpy as np
from tensorpack.dataflow.imgaug.geometry import RotationAndCropValid

from tf_pose.common import CocoPart

_network_w = 368
_network_h = 368
_scale = 2

def set_network_input_wh(w, h):
    global _network_w, _network_h
    _network_w, _network_h = w, h
def set_network_scale(scale):
    global _scale
    _scale = scale
