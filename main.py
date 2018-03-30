from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import pickle
import os, cv2
from preprocessing import parse_annotation, BatchGenerator
from utils import WeightReader, decode_netout, draw_boxes, normalize

# Set up
LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

IMAGE_H, IMAGE_W = 416, 416 # SIZE OF IMAGE
GRID_H, GRID_W   = 13, 13   # DEVICE IMAGE TO 13*13 GRIDS
BOX              = 5        # 5 ANCHOR BOX IN ONE GRID

CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')

OBJ_THRESHOLD    = 0.3      # ?
NMS_THRESHOLD    = 0.3      # ?

ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

NO_OBJECT_SCALE  = 1.0      # ?
OBJECT_SCALE     = 5.0      # ?
COORD_SCALE      = 1.0      # ?
CLASS_SCALE      = 1.0      # ?

BATCH_SIZE       = 16       # ?
WARM_UP_BATCHES  = 0        # ?
TRUE_BOX_BUFFER  = 50       # ?

pre_train_weight_path = 'yolo.weights'
train_image_folder    = '/home/vtc/git/data/train2014/'
train_annot_folder    = '/home/vtc/git/data/train2014ann/'
valid_image_folder    = '/home/vtc/git/data/valid2014/'
valid_annot_folder    = '/home/vtc/git/data/valid2014ann/'

def space_to_depth_x2(x):
    """
    the function to implement the orgnization layer
    (thanks to github.com/allanzelener/YAD2K)
    :param x:
    :return:
    """
    return tf.space_to_depth(x, block_size=2)

def custom_loss(y_true, y_pred):
    """
    Implement loss function of Yolo algorithms
    :param y_true:
    :param y_pred:
    :return:
    """
    mask_shape = tf.shape(y_true)[:4]

    cell_x = tf.to_float(
        tf.reshape(
            tf.tile(tf.range(GRID_W), [GRID_H]),
            (1, GRID_H, GRID_W, 1, 1)))

    cell_y = tf.transpose(
        cell_x,
        (0,2,1,3,4))

    cell_grid = tf.tile(
        tf.concat([cell_x, cell_y], -1),
        [BATCH_SIZE, 1, 1, 5, 1])


































