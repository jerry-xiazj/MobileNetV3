import os
import cv2
import struct
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


if __name__ == "__main__":
    img, labels = load_mnist("/mnt/d/data/mnist/", "t10k")
    with open("/home/jerry/MobileNetV3/data/train_file", "w") as wf:
        for i in range(len(labels)):
            img_i = np.reshape(img[i], (28, 28, 1))
            img_i = 255 - cv2.cvtColor(img_i, cv2.COLOR_GRAY2BGR)
            cv2.imwrite("/mnt/d/data/mnist/1/"+str(i)+".jpg", img_i)
            wf.write("/mnt/d/data/mnist/1/"+str(i)+".jpg " + str(labels[i]) + "\n")
