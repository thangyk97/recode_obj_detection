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

def get_session(gpu_fraction=0.8):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

K.set_session(get_session())

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

pre_train_weight_path = 'data/yolo.weights'
train_image_folder = '/home/vtc/git/data/train2014/'
train_annot_folder = '/home/vtc/git/recode_obj_detection/data/train2014pascal2/'
valid_image_folder = '/home/vtc/git/data/val2014/'
valid_annot_folder = '/home/vtc/git/recode_obj_detection/data/val2014pascal2/'

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

    coord_mask = tf.zeros(mask_shape) # ?
    conf_mask  = tf.zeros(mask_shape) # ?
    class_mask = tf.zeros(mask_shape) # ?

    seen       = tf.Variable(0.)      # ?
    total_recall = tf.Variable(0.)    # ?

    """
    Adjust prediction
    """
    ### adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

    ### adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])

    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    ### adjust class probabilities
    pred_box_class = y_pred[..., 5:]

    """
    Adjust groud truth
    """
    ### adjust x and y
    true_box_xy = y_true[..., 0:2] # relative position to the containing cell

    ### adjust w and h
    true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically

    ### adjust confidence
    true_wh_half = true_box_wh / 2.             # ?
    true_mins    = true_box_xy - true_wh_half   # ?
    true_maxes   = true_box_xy + true_wh_half   # ?

    pred_wh_half = pred_box_wh / 2.             # ?
    pred_mins    = pred_box_xy - pred_wh_half   # ?
    pred_maxes   = pred_box_xy + pred_wh_half   # ?

    intersect_mins  = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1] ### ?

    # Compute iou_scores
    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    true_box_conf = iou_scores * y_true[..., 4]

    ### adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)

    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE

    ### confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half       # ?

    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half       # ?

    intersect_mins  = tf.maximum(pred_mins, true_mins)                  #?
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE

    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask * y_true[..., 4] * OBJECT_SCALE

    ### class mark: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE

    """
    Warm-up training
    """
    no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE / 2.)
    seen = tf.assign_add(seen, 1.)

    true_box_xy, true_box_wh, coord_mask = \
        tf.cond(tf.less(seen, WARM_UP_BATCHES),
                lambda : [
                    true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                    true_box_wh + tf.ones_like(true_box_wh) * np.reshape(ANCHORS, [1,1,1,BOX,2]) * no_boxes_mask,
                    tf.ones_like(coord_mask)],
                lambda : [
                    true_box_xy,
                    true_box_wh,
                    coord_mask])
    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) \
              / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) \
              / (nb_coord_box + 1e-6) / 2.
    loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * coord_mask) \
              / (nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=true_box_class,
                    logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    loss = loss_xy + loss_wh + loss_conf + loss_class

    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(
        tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    """
    Debugging code
    """
    current_recall = nb_pred_box / (nb_true_box + 1e-6)
    total_recall   = tf.assign_add(total_recall, current_recall)

    loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Confidence \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

    return loss

"""
Construct the network
"""
input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))

### Layer 1
x = Conv2D(32, (3,3), strides=(1,1), padding='same',
           name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2,2))(x)

### Layer 2
x = Conv2D(64, (3,3), strides=(1,1), padding='same',
           name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2,2))(x)

### Layer 3
x = Conv2D(128, (3,3), strides=(1,1), padding='same',
           name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 4
x = Conv2D(64, (1,1), strides=(1,1), padding='same',
           name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 5
x = Conv2D(128, (3,3), strides=(1,1), padding='same',
           name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2,2))(x)

### Layer 6
x = Conv2D(256, (3,3), strides=(1,1), padding='same',
           name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 7
x = Conv2D(128, (1,1), strides=(1,1), padding='same',
           name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 8
x = Conv2D(256, (3,3), strides=(1,1), padding='same',
           name='conv_8', use_bias=False)(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

### Layer 9
x = Conv2D(512, (3,3), strides=(1,1), padding='same',
           name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 10
x = Conv2D(256, (1,1), strides=(1,1), padding='same',
           name='conv_10', use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 11
x = Conv2D(512, (3,3), strides=(1,1), padding='same',
           name='conv_11', use_bias=False)(x)
x = BatchNormalization(name='norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 12
x = Conv2D(256, (1,1), strides=(1,1), padding='same',
           name='conv_12', use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 13
x = Conv2D(512, (3,3), strides=(1,1), padding='same',
           name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x #########
x = MaxPooling2D(pool_size=(2,2))(x)

### Layer 14
x = Conv2D(1024, (3,3), strides=(1,1), padding='same',
           name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 15
x = Conv2D(512, (1,1), strides=(1,1), padding='same',
           name='conv_15', use_bias=False)(x)
x = BatchNormalization(name='norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 16
x = Conv2D(1024, (3,3), strides=(1,1), padding='same',
           name='conv_16', use_bias=False)(x)
x = BatchNormalization(name='norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 17
x = Conv2D(512, (1,1), strides=(1,1), padding='same',
           name='conv_17', use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 18
x = Conv2D(1024, (3,3), strides=(1,1), padding='same',
           name='conv_18', use_bias=False)(x)
x = BatchNormalization(name='norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 19
x = Conv2D(1024, (3,3), strides=(1,1), padding='same',
           name='conv_19', use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 20
x = Conv2D(1024, (3,3), strides=(1,1), padding='same',
           name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 21
skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same',
                         name='conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)     # ?

x = concatenate([skip_connection, x])
### Layer 22
x = Conv2D(1024, (3,3), strides=(1,1), padding='same',
           name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)

### Layer 23
x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1),
           padding='same', name='conv_23')(x)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

# small hack to allow true_boxes to be registered when Keras build the model
# for more information: https://github.com/fchollet/keras/issues/2790
output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([input_image, true_boxes], output)

model.summary()

"""
Load pretrained weights
load the weights originally provided by YOLO
"""
weight_reader = WeightReader(pre_train_weight_path)
weight_reader.reset()
nb_conv = 23

for i in range(1, nb_conv + 1):
    conv_layer = model.get_layer('conv_' + str(i))

    if i < nb_conv:
        norm_layer = model.get_layer('norm_' + str(i))
        size       = np.prod(norm_layer.get_weights()[0].shape)

        beta       = weight_reader.read_bytes(size)
        gamma      = weight_reader.read_bytes(size)
        mean       = weight_reader.read_bytes(size)
        var        = weight_reader.read_bytes(size)

        weights    = norm_layer.set_weights([gamma, beta, mean, var])

    if len(conv_layer.get_weights()) > 1:
        bias    = weight_reader.read_bytes(
            np.prod(conv_layer.get_weights()[1].shape))
        kernel  = weight_reader.read_bytes(
            np.prod(conv_layer.get_weights()[0].shape))
        kernel  = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel  = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(
            np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(
            list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel])

"""Randomize weights of the last year"""
layer   = model.layers[-4] # the last conv layer
weights = layer.get_weights()

new_kernel  = np.random.normal(size=weights[0].shape) / (GRID_W*GRID_H)
new_bias    = np.random.normal(size=weights[1].shape) / (GRID_H*GRID_W)

layer.set_weights([new_kernel, new_bias])

"""Parse the annotations to construct train generator and validation generator"""
generator_config = {
    'IMAGE_H'           : IMAGE_H,
    'IMAGE_W'           : IMAGE_W,
    'GRID_H'            : GRID_H,
    'GRID_W'            : GRID_W,
    'BOX'               : BOX,
    'LABELS'            : LABELS,
    'CLASS'             : len(LABELS),
    'ANCHORS'           : ANCHORS,
    'BATCH_SIZE'        : BATCH_SIZE,
    'TRUE_BOX_BUFFER'   : 50,
}

train_imgs, seen_train_labels = parse_annotation(
    train_annot_folder,
    train_image_folder,
    labels=LABELS)

train_batch = BatchGenerator(train_imgs, generator_config, norm=normalize)

valid_imgs, seen_valid_labels = parse_annotation(
    valid_annot_folder,
    valid_image_folder,
    labels=LABELS)

valid_batch = BatchGenerator(valid_imgs, generator_config, norm=normalize, jitter=False)

"""Setup a few callbacks and start the training"""
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=3,
    mode='min',
    verbose=1)

checkpoint = ModelCheckpoint(
    filepath='weights_coco.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    period=1)

tb_counter  = len([log for log in os.listdir(os.path.expanduser('~/logs/')) if 'coco_' in log]) + 1

tensorboard = TensorBoard(
    log_dir=os.path.expanduser('~/logs/') + 'coco_' + '_' + str(tb_counter),
    histogram_freq=0,
    write_graph=True,
    write_images=False
)

optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_loss, optimizer=optimizer)

model.fit_generator(
    generator        = train_batch,
    steps_per_epoch  = len(train_batch),
    epochs           = 1,
    verbose          = 1,
    validation_data  = valid_batch,
    validation_steps = len(valid_batch),
    callbacks        = [early_stop, checkpoint, tensorboard],
    max_queue_size   = 3)










































