"""Provides a WGan for MNIST data."""
import os
import tensorflow as tf
from gan_tf.w_gan import WGan

n_z = 10
n_h = 128
image_length = 28
n_x = image_length**2


def _get_critic_logits(sample):
    x = sample
    initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    x = tf.layers.dense(
        x, n_h, activation=tf.nn.relu, kernel_initializer=initializer)
    x = tf.layers.dense(
        x, 1, activation=None, kernel_initializer=initializer)

    logits = tf.squeeze(x, 1)
    return logits


def _get_generator_sample(features):
    x = features
    initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    x = tf.layers.dense(
        x, n_h, activation=tf.nn.relu, kernel_initializer=initializer)
    x = tf.layers.dense(
        x, n_x, activation=tf.nn.sigmoid,
        kernel_initializer=initializer)
    tf.summary.image('generated_image', tf.reshape(x, (-1, 28, 28, 1)))
    return x


_model_dir = os.path.join(os.path.dirname(__file__), 'WGan_model')
mnist_gan = WGan(_get_generator_sample, _get_critic_logits, _model_dir)
