"""Training script for GAN."""
import tensorflow as tf
from gan import pose_gan, get_random_generator_input, get_real_sample


def train_pose_gan(gan_id, config=None, **train_args):
    """
    Train the specified model.

    `train_args` should have `steps` or `max_steps`.
    """
    gan = pose_gan(gan_id, config=config)
    return gan.train(get_random_generator_input, get_real_sample, **train_args)


if __name__ == '__main__':
    import argparse
    from serialization import register_defaults
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'gan_id',
        help='id of GAN spec defined in gan_params')
    parser.add_argument(
        '-s', '--max_steps', type=float, default=1e7,
        help='maximum number of steps to train until')
    args = parser.parse_args()
    register_defaults()
    tf.logging.set_verbosity(tf.logging.INFO)
    train_pose_gan(args.gan_id, max_steps=int(args.max_steps))
