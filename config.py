# Author: Jerry Xia <jerry_xiazj@outlook.com>

from easydict import EasyDict as edict


CFG = edict()
# config train
CFG.data_aug = False
CFG.train_epoch = 60
CFG.batch_size = 5
CFG.batch_per_epoch = 20
CFG.lr_init = 0.1
CFG.lr_decay = 0.09
CFG.decay_step = 3 * CFG.batch_per_epoch
# config path
CFG.train_file = "./data/train_file"
CFG.val_file = "./data/val_file"
CFG.log_dir = "./log/"
CFG.checkpoint_dir = CFG.log_dir + "ckpt/"
CFG.checkpoint_prefix = CFG.checkpoint_dir + "ckpt"
# config data
CFG.classes = ["OO", "YY", "Other"]
CFG.num_classes = len(CFG.classes)
# config model
CFG.input_shape = [224, 224]
