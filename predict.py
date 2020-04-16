# Author: Jerry Xia <jerry_xiazj@outlook.com>

import time
import cv2
import tensorflow as tf
from config import CFG
from core.model import MobileNetv3_small
from core.dataset import Dataset


def draw_class(img, classes, score, name):
    bbox_mess = '%s: %.2f' % (classes, score)
    cv2.putText(img, bbox_mess, (0, 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(CFG.log_dir+name, img)


tf.keras.backend.set_learning_phase(False)

####################################
#          Generate Dataset        #
####################################
test_set = Dataset(CFG.test_file, CFG.batch_size, 1)

####################################
#           Create Model           #
####################################
tf.print("Start creating model.")
input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))
output_tensor = MobileNetv3_small(CFG.num_classes)(input_tensor)
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

ckpt = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(ckpt, CFG.checkpoint_dir, max_to_keep=3)
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    tf.print("Restored from ", manager.latest_checkpoint)
else:
    tf.print("Initializing from scratch.")
tf.print("Finish creating model.")

####################################
#             Predict              #
####################################

start = time.time()
global_step = 0

for _, val_img, val_class in test_set:
    pred = model(val_img)

# cv2.imwrite(CFG.log_dir+"1.png", val_img[0]*255)

index = tf.argmax(pred, axis=-1).numpy()
for i in range(CFG.batch_size):
    ind = index[i]
    classes = CFG.classes[ind]
    score = pred[i, ind].numpy()
    img = val_img[i]*255
    draw_class(img, classes, score, str(i)+".png")
