# Author: Jerry Xia <jerry_xiazj@outlook.com>

from easydict import EasyDict as edict


CFG = edict()
# config train
CFG.data_aug = False
CFG.train_epoch = 90
CFG.batch_size = 10
CFG.batch_per_epoch = 9
CFG.lr_init = 0.001
CFG.lr_decay = 0.9
CFG.decay_step = 3 * CFG.batch_per_epoch
# config path
CFG.train_file = "./data/person_train_file"
CFG.val_file = "./data/person_val_file"
CFG.test_file = "./data/person_val_file"
CFG.log_dir = "./log/"
CFG.checkpoint_dir = CFG.log_dir + "ckpt_person/"
# config data
# CFG.classes = ["person", "bird", "cat", "cow", "dog", "horse",
#                "sheep", "aeroplane", "bicycle", "boat", "bus",
#                "car", "motorbike", "train", "bottle", "chair",
#                "diningtable", "pottedplant", "sofa", "tvmonitor"]
# CFG.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
CFG.classes = ["YY", "OO"]
CFG.num_classes = len(CFG.classes)
# config model
CFG.input_shape = [224, 224]
