"""Training script for GAN."""
import tensorflow as tf
from gan import GanBuilder


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

    builder = GanBuilder(args.gan_id)

    def train_pose_gan(gan_id, config=None, **train_args):
        """
        Train the specified model.

        `train_args` should have `steps` or `max_steps`.
        """
        def input_fn():
            features = builder.get_random_generator_input()
            labels = builder.get_real_sample()
            return features, labels

        gan = builder.gan_estimator()
        return gan.train(input_fn, **train_args)

    tf.logging.set_verbosity(tf.logging.INFO)
    train_pose_gan(args.gan_id, max_steps=int(args.max_steps))
