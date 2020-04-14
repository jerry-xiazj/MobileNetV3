# Author: Jerry Xia <jerry_xiazj@outlook.com>

import os
import cv2
import random
import numpy as np
import tensorflow as tf
from config import CFG


class Dataset:

    def __init__(self, file_path, batch_size, batch_per_epoch):
        self.ann_lines, self.num_samples = self.load_annotations(file_path)
        self.batch_count = 0
        self.batch_size = batch_size
        self.batch_per_epoch = batch_per_epoch

    def __iter__(self):
        return self

    def __len__(self):
        return self.batch_per_epoch

    def __next__(self):
        with tf.device('/cpu:0'):
            batch_image = np.zeros(
                (self.batch_size, CFG.input_shape[0], CFG.input_shape[1], 3), dtype=np.float32
            )
            batch_class = np.zeros(
                (self.batch_size, CFG.num_classes), dtype=np.float32
            )
            if self.batch_count < self.batch_per_epoch:
                num = 0
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index %= self.num_samples
                        np.random.shuffle(self.ann_lines)
                    ann_line = self.ann_lines[index]
                    image, category = self.parse_annotation(ann_line)
                    batch_image[num, :, :, :] = image
                    batch_class[num, :] = np.eye(CFG.num_classes)[category]
                    num += 1
                self.batch_count += 1
                return self.batch_count, batch_image, batch_class
            else:
                self.batch_count = 0
                raise StopIteration

    def load_annotations(self, file_path):
        with open(file_path, 'r') as rf:
            ann_lines = rf.readlines()
        ann_lines = [ann.rstrip('\n') for ann in ann_lines]
        return ann_lines, len(ann_lines)

    def random_horizontal_flip(self, image):
        if random.random() < 0.5:
            image = image[:, ::-1, :]
        return image

    def resize_img(self, img, input_shape):
        """
        Resize the image according to the aspect ratio, fill the blank area with (128, 128, 128).
        """
        img_h, img_w, _ = img.shape
        inp_h, inp_w = input_shape
        new_w = int(img_w * min(inp_w/img_w, inp_h/img_h))
        new_h = int(img_h * min(inp_w/img_w, inp_h/img_h))
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        img_data = np.full(shape=[inp_h, inp_w, 3], fill_value=128.0)
        dw, dh = (inp_w-new_w)//2, (inp_h-new_h)//2
        img_data[dh:dh+new_h, dw:dw+new_w, :] = resized_img
        img_data = img_data / 255.
        return img_data

    def parse_annotation(self, ann_line):
        """Parse annotation file.
        ann_line: one line of annotation file
        """
        ann_line = ann_line.split(' ')
        if not os.path.exists(ann_line[0]):
            raise KeyError("%s does not exist ... " % ann_line[0])
        image = cv2.cvtColor(cv2.imread(ann_line[0]), cv2.COLOR_BGR2RGB)
        if CFG.data_aug:
            image = self.random_horizontal_flip(image)
        image = self.resize_img(image, CFG.input_shape)
        return image, ann_line[1]
