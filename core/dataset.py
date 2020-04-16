# Author: Jerry Xia <jerry_xiazj@outlook.com>

import os
import cv2
import numpy as np
import tensorflow as tf
from config import CFG
from core.utils import random_horizontal_flip, resize_img


class Dataset:

    def __init__(self, file_path, batch_size, batch_per_epoch, train=True):
        self.ann_lines, self.num_samples = self._load_annotation(file_path)
        self.batch_count = 0
        self.batch_size = batch_size
        self.batch_per_epoch = batch_per_epoch
        self.train = train

    def __iter__(self):
        return self

    def __len__(self):
        return self.batch_per_epoch

    def __next__(self):
        with tf.device('/cpu:0'):
            if self.batch_count < self.batch_per_epoch:
                self.batch_count += 1
                return self.batch_count, self.generate_batch()
            else:
                self.batch_count = 0
                raise StopIteration

    def _load_annotation(self, file_path):
        with open(file_path, 'r') as rf:
            ann_lines = rf.readlines()
        ann_lines = [ann.rstrip('\n') for ann in ann_lines]
        # np.random.shuffle(ann_lines)
        return ann_lines, len(ann_lines)

    def _parse_annotation(self, ann_line):
        """Parse annotation file.
        ann_line: one line of annotation file
        """
        ann_line = ann_line.split(' ')
        if not os.path.exists(ann_line[0]):
            raise KeyError("%s does not exist ... " % ann_line[0])
        image = cv2.cvtColor(cv2.imread(ann_line[0]), cv2.COLOR_BGR2RGB)
        if CFG.data_aug:
            image = random_horizontal_flip(image)
        image = resize_img(image, CFG.input_shape)
        if self.train:
            return image, int(ann_line[1])
        else:
            return image

    def generate_batch(self):
        batch_image = np.zeros(
            (self.batch_size, CFG.input_shape[0], CFG.input_shape[1], 3), dtype=np.float32
        )
        if self.train:
            batch_class = np.zeros(
                (self.batch_size, CFG.num_classes), dtype=np.float32
            )
        num = 0
        while num < self.batch_size:
            index = self.batch_count * self.batch_size + num
            if index >= self.num_samples:
                index %= self.num_samples
                np.random.shuffle(self.ann_lines)
            ann_line = self.ann_lines[index]
            res = self._parse_annotation(ann_line)
            if self.train:
                batch_image[num, :, :, :] = res[0]
                batch_class[num, :] = np.eye(CFG.num_classes)[res[1]]
            else:
                batch_image[num, :, :, :] = res
            num += 1
        if self.train:
            return batch_image, batch_class
        else:
            return batch_image

    def generate_origin(self):
        for line in self.ann_lines:
            img = cv2.imread(line.split(' ')[0])
            yield img
