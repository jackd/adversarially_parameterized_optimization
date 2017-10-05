import tensorflow as tf
from model import mnist_gan, n_z, image_length
import matplotlib.pyplot as plt

batch_size = 32


def generator_input_fn():
    return tf.random_uniform((batch_size, n_z), -1., 1., dtype=tf.float32)


cmap = plt.get_cmap('gray')
print('Images will generate forever: ctrl + c to stop')
for sample in mnist_gan.generate(generator_input_fn, iterate_batches=False):
    plt.imshow(sample.reshape(image_length, image_length), cmap=cmap)
    plt.show()
