"""Example script for training WGan on mnist data."""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import mnist_gan, n_z, n_x


if __name__ == '__main__':
    import numpy as np

    batch_size = 32
    num_threads = 8

    def get_generator_input():
        return tf.random_uniform((batch_size, n_z), -1., 1., dtype=tf.float32)

    def get_real_sample():
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        mnist = input_data.read_data_sets(data_dir, one_hot=True)

        images = mnist.train.images
        assert(images.shape[1] == n_x)
        assert(np.max(images) == 1)
        image, = tf.train.slice_input_producer(
            [tf.constant(images, dtype=tf.float32)], shuffle=True)
        # image = tf.image.per_image_standardization(image)
        image_batch = tf.train.batch(
            [image], batch_size=batch_size, num_threads=num_threads)

        tf.summary.image(
            'real_image', tf.reshape(image_batch, (-1, 28, 28, 1)))

        return image_batch

    mnist_gan.train(get_generator_input, get_real_sample, max_steps=1000000)
