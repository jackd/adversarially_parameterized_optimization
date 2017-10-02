"""Training script for GAN."""
import tensorflow as tf
from human_pose_util.register import register_defaults
from gan import pose_gan, get_random_generator_input, get_real_sample


def train_pose_gan(model_name, config=None, **train_args):
    """
    Train the specified model.

    `train_args` should have `steps` or `max_steps`.
    """
    gan = pose_gan(model_name, config=config)
    return gan.train(get_random_generator_input, get_real_sample, **train_args)


if __name__ == '__main__':
    model_name = 'small'
    max_steps = int(1e7)
    tf.logging.set_verbosity(tf.logging.INFO)
    register_defaults()
    train_pose_gan(model_name, max_steps=max_steps)
