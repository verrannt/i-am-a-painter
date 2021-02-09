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

    def load_dataset(self, filenames, labeled=True, ordered=False, augment=False):
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=AUTOTUNE)
        if augment:
            dataset = dataset.map(self.augment_image, num_parallel_calls=AUTOTUNE)
            dataset = dataset.repeat(count=2)
        return dataset