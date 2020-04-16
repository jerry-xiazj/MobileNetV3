# Author: Jerry Xia <jerry_xiazj@outlook.com>

import time
import tensorflow as tf
from config import CFG
from core.model import MobileNetv3_small
from core.dataset import Dataset


tf.keras.backend.set_learning_phase(True)

####################################
#          Generate Dataset        #
####################################
train_set = Dataset(CFG.train_file, CFG.batch_size, CFG.batch_per_epoch)
val_set = Dataset(CFG.val_file, CFG.batch_size, 1)

####################################
#           Create Model           #
####################################
tf.print("Start creating model.")
input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))
output_tensor = MobileNetv3_small(CFG.num_classes)(input_tensor)
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

optimizer = tf.keras.optimizers.RMSprop(lr=CFG.lr_init, momentum=0.9)
cce = tf.keras.losses.CategoricalCrossentropy()
avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
# ema = tf.train.ExponentialMovingAverage(decay=0.9999)

ckpt = tf.train.Checkpoint(model=model)
# ckpt = tf.train.Checkpoint(ema.variables_to_restore(), model=model)
manager = tf.train.CheckpointManager(ckpt, CFG.checkpoint_dir, max_to_keep=3)
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    tf.print("Restored from ", manager.latest_checkpoint)
else:
    tf.print("Initializing from scratch.")
tf.print("Finish creating model.")

####################################
#              Train               #
####################################


@tf.function
def train_step(img, classes):
    with tf.GradientTape() as tape:
        pred = model(img)
        loss = cce(classes, pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # ema.apply(model.trainable_variables)
    return loss


tf.print("Start training for", CFG.train_epoch, "epochs.")
tf.print("Batch per epoch:", CFG.batch_per_epoch, "Batch size:", CFG.batch_size)
start = time.time()
global_step = 0

for epoch in range(1, 1+CFG.train_epoch):

    for batch, train_img, train_class in train_set:
        tf.print(
            "=> Epoch: %3d" % epoch, "/", CFG.train_epoch,
            "Batch: %3d" % batch, "/", CFG.batch_per_epoch,
            "lr: %.5e" % optimizer.lr.numpy(), end=" "
        )
        loss_train = train_step(train_img, train_class)
        tf.print("loss_train: %.5f" % loss_train)
        avg_loss.update_state(loss_train)

        global_step += 1
        if global_step % CFG.decay_step == 0:
            learning_rate = optimizer.lr.numpy() * CFG.lr_decay
            optimizer.lr.assign(learning_rate)

    for _, val_img, val_class in val_set:
        pred = model(val_img)
        loss_val = cce(val_class, pred)
        avg_val_loss.update_state(loss_val)

    tf.print("Average train loss:      %.5f" % avg_loss.result().numpy())
    tf.print("Average validation loss: %.5f" % avg_val_loss.result().numpy())

    if epoch == 1:
        n_loss_raise = 0
        val_loss_last = avg_val_loss.result().numpy()
    elif avg_val_loss.result().numpy() < val_loss_last:
        n_loss_raise = 0
        val_loss_last = avg_val_loss.result().numpy()
    elif n_loss_raise < 2:
        n_loss_raise += 1
        val_loss_last = avg_val_loss.result().numpy()
        avg_loss.reset_states()
        avg_val_loss.reset_states()
        tf.print("Validation loss raise:", n_loss_raise, "/ 3")
        continue
    else:
        avg_loss.reset_states()
        avg_val_loss.reset_states()
        tf.print("Early stop.")
        break

    avg_loss.reset_states()
    avg_val_loss.reset_states()
    save_path = manager.save()
    tf.print("Saved checkpoint for epoch", epoch, ":", save_path)

tf.print("Finish training. Time taken: %.2f" % time.time()-start), "h.")
