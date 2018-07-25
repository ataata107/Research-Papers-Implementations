import matplotlib as mpl
mpl.use('Agg')      # training mode, no screen should be open. (It will block training loop)


import argparse
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorpack.dataflow.remote import RemoteDataZMQ

from pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoPose
from pose_augment import set_network_input_wh, set_network_scale
from common import get_sample_images
from networks import get_network

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='personlab_resnet101', help='model name')
    parser.add_argument('--datapath', type=str, default='/data/public/rw/coco/annotations')
    parser.add_argument('--imgpath', type=str, default='/data/public/rw/coco/')
    parser.add_argument('--batchsize', type=int, default=96)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max-epoch', type=int, default=30)
    parser.add_argument('--lr', type=str, default='0.01')
    parser.add_argument('--modelpath', type=str, default='/data/private/tf-openpose-models-2018-2/')
    parser.add_argument('--logpath', type=str, default='/data/private/tf-openpose-log-2018-2/')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--remote-data', type=str, default='', help='eg. tcp://0.0.0.0:1027')

    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    args = parser.parse_args()

    if args.gpus <= 0:
        raise Exception('gpus <= 0')

    # define input placeholder
    set_network_input_wh(args.input_width, args.input_height)
    scale = 4

    if args.model in ['cmu', 'vgg', 'mobilenet_thin', 'mobilenet_try', 'mobilenet_try2', 'mobilenet_try3', 'hybridnet_try']:
        scale = 8

    set_network_scale(scale)
    output_w, output_h = args.input_width // scale, args.input_height // scale

    logger.info('define model+')
     with tf.device(tf.DeviceSpec(device_type="CPU")):
         input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.input_height, args.input_width, 3), name='image')
         vectmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, 38), name='vectmap')
         heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, 19), name='heatmap')

          # prepare data
          if not args.remote_data:
              df = get_dataflow_batch(args.datapath, True, args.batchsize, img_path=args.imgpath)
              
