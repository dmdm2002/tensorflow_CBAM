import os
import glob
import tensorflow as tf
import keras
import numpy as np


class Loader(object):
    def __init__(self, root, classes):
        self.root = root
        path_list = []
        labels = []

        class_indexing_list = np.arange(len(classes))

        for label in classes:
            temp = []
            list = glob.glob(f'{root}/{label}/*.png')

            for i in range(len(list)):
                temp.append(class_indexing_list[i])

            path_list = path_list + list
            labels = labels + temp

        path_list = np.array(path_list)
        labels = np.array(labels)

        label_list = tf.one_hot(labels, depth=len(classes))

        self.path_ds = tf.data.Dataset.from_tensor_slices(path_list)
        self.label_ds = tf.data.Dataset.from_tensor_slices(label_list)

    def decode_img(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, 3)
        img = tf.image.resize(img, [224, 224]) / 255.

        return img

    def load(self):
        img_ds = self.path_ds.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = tf.data.Dataset.zip(img_ds, self.label_ds)

        return ds


def configure_for_performance(ds, cnt, shuffle=False):
    if shuffle==True:
        ds = ds.shuffle(buffer_size=cnt)
        ds = ds.batch(1)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    elif shuffle==False:
        ds = ds.batch(1)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds