import os
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
import xml.etree.ElementTree as ET
from utils import BoundBox, normalize, bbox_iou

def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs    = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.next)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attribute in list(elem):
                    if 'name' in attribute.tag:
                        obj['name'] = attribute.text
                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                    
                    if 'bndbox' in attribute.tag:
                        for dim in list(attribute):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels

class BatchGenerator(Sequence):
    def __init__(
        self, images, config, shuffle=True, jitter=True, norm=None):

        self.generator = None
        self.images    = images
        self.config    = config
        self.shuffle   = shuffle
        self.jitter    = jitter
        self.norm      = norm

        self.anchors   = [
            BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i + 1])
            for i in range(int(len(config['ANCHORS'])) // 2 )]
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        