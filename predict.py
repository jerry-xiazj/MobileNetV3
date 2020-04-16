# Author: Jerry Xia <jerry_xiazj@outlook.com>

import time
import tensorflow as tf
from config import CFG
from core.utils import draw_class
from core.model import MobileNetv3_small
from core.dataset import Dataset


tf.keras.backend.set_learning_phase(False)

####################################
#          Generate Dataset        #
####################################
test_set = Dataset(CFG.test_file, CFG.batch_size, 1, train=False)

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

test_img = test_set.generate_batch()
pred = model(test_img)
index = tf.argmax(pred, axis=-1).numpy()

for i, img in enumerate(test_set.generate_origin()):
    classes = CFG.classes[index[i]]
    print(classes)
    score = pred[i, index[i]].numpy()
    print(score)
    path = CFG.log_dir + "test/" + str(i)+".png"
    draw_class(img, classes, score, path)

tf.print("Finish predicting. Time taken:", time.time()-start, "sec.")
