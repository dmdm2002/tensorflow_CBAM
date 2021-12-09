import tensorflow as tf
import keras
from CBAM_ResNet import ResNet
from DataLoader import Loader

class setting_option(object):
    def __init__(self):
        self.Loader = Loader
        self.epochs = 10
        self.batchSZ = 2
        self.train_root = 'E:/Iris_dataset/nd_labeling_iris_data/Proposed/1-fold/A/iris'
        self.test_root = 'E:/Iris_dataset/nd_labeling_iris_data/Proposed/1-fold/B/iris'
        self.val_root = 'E:/Iris_dataset/nd_labeling_iris_data/Proposed/1-fold/B/iris'
        self.class_list = ['fake', 'live']
        self.input_shape = (224, 224, 3)
        self.depth = 50
        self.num_class = 2
        self.traincnt = 4554
        self.testcnt = 5018
        self.valcnt = 5018
        self.train_log_dir = './log/fit/temp'
        self.ckp_path = './temp_ckp'
        # self.input_shape = (224, 224, 3)
        # self.depth =
        # self.Model = ResNet()()
        # self.loos_fn =
        # self.Model = ResNet()(input_shape, depth, classes)