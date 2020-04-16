# Author: Jerry Xia <jerry_xiazj@outlook.com>

from easydict import EasyDict as edict


CFG = edict()
# config train
CFG.data_aug = False
CFG.train_epoch = 9
CFG.batch_size = 10
CFG.batch_per_epoch = 1000
CFG.lr_init = 0.0001
CFG.lr_decay = 0.09
CFG.decay_step = 3 * CFG.batch_per_epoch
# config path
CFG.train_file = "./data/train_file"
CFG.val_file = "./data/val_file"
CFG.test_file = "./data/test_file"
CFG.log_dir = "./log/"
CFG.checkpoint_dir = CFG.log_dir + "ckpt/"
CFG.checkpoint_prefix = CFG.checkpoint_dir + "ckpt"
# config data
<<<<<<< HEAD
# CFG.classes = ["person", "bird", "cat", "cow", "dog", "horse",
#                "sheep", "aeroplane", "bicycle", "boat", "bus",
#                "car", "motorbike", "train", "bottle", "chair",
#                "diningtable", "pottedplant", "sofa", "tvmonitor"]
CFG.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
=======
# CFG.classes = ["YY", "OO", "other"]
CFG.classes = ["person", "bird", "cat", "cow", "dog", "horse",
               "sheep", "aeroplane", "bicycle", "boat", "bus",
               "car", "motorbike", "train", "bottle", "chair",
               "diningtable", "pottedplant", "sofa", "tvmonitor"]
>>>>>>> 96f0b778e570a47dd5fa84508d52deb12ef0618a
CFG.num_classes = len(CFG.classes)
# config model
CFG.input_shape = [224, 224]
