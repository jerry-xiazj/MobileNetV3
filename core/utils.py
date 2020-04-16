import cv2
import random
import numpy as np
from config import CFG


def random_horizontal_flip(image):
    if random.random() < 0.5:
        image = image[:, ::-1, :]
    return image


def resize_img(img, input_shape):
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


def draw_class(img, classes, score, path):
    image_h, image_w, _ = img.shape
    word = '%s: %.2f' % (classes, score)
    thick = int(0.6 * (image_h + image_w) / 600)
    cv2.putText(img, word, (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                3, (0, 0, 0), thick//2, lineType=cv2.LINE_AA)
    cv2.imwrite(path, img)
