import os
import re

import numpy as np
import tensorflow as tf

class DataLoader():

    def __init__(self, image_size):
        self.image_size = image_size

    def decode_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        image = tf.reshape(image, [*self.image_size, 3])
        return image

    def read_tfrecord(self, example):
        tfrecord_format = {
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(example, tfrecord_format)
        image = self.decode_image(example['image'])
        return image

    def augment_image(self, image):
        flip_image = tf.image.random_flip_left_right(image)
        return flip_image

    def load_dataset(
            self, 
            filenames, 
            batch_size,
            vsplit=0.0,
            shuffle=False,
            augment=False):
        
        assert 0. <= vsplit < 1., \
            "Validation split must be between 0. and 1."

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        n_entries = count_data_items(filenames)

        dataset = tf.data.TFRecordDataset(filenames)
        
        if shuffle:
            dataset = dataset.shuffle(1234)
        
        if vsplit:
            trainset = dataset.take((1.-vsplit) * n_entries)
            valset = dataset.skip((1.-vsplit) * n_entries)
            valset = valset.map(self.read_tfrecord, num_parallel_calls=AUTOTUNE)
        else:
            trainset = dataset

        trainset = trainset.map(self.read_tfrecord, num_parallel_calls=AUTOTUNE)
                
        if augment:
            trainset = trainset.map(self.augment_image, num_parallel_calls=AUTOTUNE)
            trainset = trainset.repeat(count=2)

        if vsplit:
            return trainset.batch(batch_size), valset.batch(batch_size)
        else:
            return trainset.batch(batch_size)

def get_filenames(use_tpus=False, data_path='/kaggle/input/gan-getting-started/'):

    if use_tpus:
        from kaggle import KaggleDatasets
        GCS_PATH =  KaggleDatasets().get_gcs_path()

        MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))
        PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))
        
    else:
        MONET_TFREC_PATH = data_path + 'monet_tfrec/'
        PHOTO_TFREC_PATH = data_path + 'photo_tfrec/'

        MONET_FILENAMES = [MONET_TFREC_PATH + filename for filename in os.listdir(MONET_TFREC_PATH)]
        PHOTO_FILENAMES = [PHOTO_TFREC_PATH + filename for filename in os.listdir(PHOTO_TFREC_PATH)]

    print('Monet TFRecord Files:', len(MONET_FILENAMES))
    print('Photo TFRecord Files:', len(PHOTO_FILENAMES))

    n_monet_samples = count_data_items(MONET_FILENAMES)
    n_photo_samples = count_data_items(PHOTO_FILENAMES)

    print('Monet Samples : ',n_monet_samples)
    print('Photo Samples : ',n_photo_samples)

    return MONET_FILENAMES, PHOTO_FILENAMES

def count_data_items(filenames):
    n = [int(re.compile(r'-([0-9]*)\.').search(filename).group(1)) for filename in filenames]
    return np.sum(n)