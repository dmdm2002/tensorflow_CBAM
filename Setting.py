import tensorflow as tf
import keras
from CBAM_ResNet import ResNet
from DataLoader import Loader

class setting_option(object):
    def __init__(self):
        self.Loader = Loader
        self.epochs = 50
        self.batchSZ = 2
        self.train_root = ''
        self.test_root = ''
        self.val_root = ''
        self.class_list = []
        self.input_shape = (224, 224, 3)
        self.depth = 50
        self.num_class = 2
        self.traincnt = 0
        self.testcnt = 0
        self.valcnt = 0
        self.train_log_dir = ''
        self.ckp_path = ''
        # self.input_shape = (224, 224, 3)
        # self.depth =
        # self.Model = ResNet()()
        # self.loos_fn =
        # self.Model = ResNet()(input_shape, depth, classes)