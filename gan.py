"""Provides GAN for parameterization/feasibility loss learning."""
from __future__ import division
import tensorflow as tf
import tensorflow.contrib.gan as tfgan
from human_pose_util.register import get_skeleton, get_dataset
from human_pose_util.dataset.normalize import \
    filter_dataset, normalize_dataset, dataset_to_p3w
from human_pose_util.dataset.group import copy_group
from adversarially_parameterized_optimization.serialization import \
    load_gan_params, gan_model_dir


_activations = {
    'relu': tf.nn.relu,
    'elu': tf.nn.elu
}

_generator_losses = {
    'minimax': tfgan.losses.minimax_generator_loss,
    'modified': tfgan.losses.modified_generator_loss,
    'wasserstein': tfgan.losses.wasserstein_generator_loss,
    'acgan': tfgan.losses.acgan_generator_loss,
    'least_squares': tfgan.losses.least_squares_generator_loss,
}

_discriminator_losses = {
    'minimax': tfgan.losses.minimax_discriminator_loss,
    'modified': tfgan.losses.modified_discriminator_loss,
    'wasserstein': tfgan.losses.wasserstein_discriminator_loss,
    'acgan': tfgan.losses.acgan_discriminator_loss,
    'least_squares': tfgan.losses.least_squares_discriminator_loss,
}


class GanBuilder(object):
    """Builder class for GANEstimator (and other related utilities)."""

    def __init__(self, gan_id):
        """Build with a gan_id defined in gan_params/gan_id.json."""
        self.id = gan_id
        self.params = load_gan_params(gan_id)

    def _get_n_joints(self):
        skeleton_id = self.params['dataset']['normalize_kwargs']['skeleton_id']
        return get_skeleton(skeleton_id).n_joints

    def get_generator_sample(self, features):
        """Get the generator sample."""
        n_g = self.params['n_g']
        n_joints = self._get_n_joints()
        x = features
        with tf.variable_scope('generator'):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            for i, n_h in enumerate(n_g):
                x = tf.layers.dense(
                    x, n_h, activation=_activations[self.params['activation']],
                    kernel_initializer=initializer,
                    name='dense%d' % i)
            x = tf.layers.dense(
                x, n_joints * 3, activation=None,
                kernel_initializer=initializer,
                name='dense_final')
            shape = x.shape.as_list()[:-1] + [n_joints, 3]
            for i, s in enumerate(shape):
                if s is None:
                    shape[i] = -1
                    break
            x = tf.reshape(x, shape)
        return x

    def get_critic_logits(self, sample, generator_inputs):
        """Critic logits function used in GAN."""
        n_joints = self._get_n_joints()
        n_c = self.params['n_c']
        with tf.variable_scope('critic'):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            shape = sample.shape.as_list()[:-2]
            for i, s in enumerate(shape):
                if s is None:
                    shape[i] = -1
                    break
            shape.append(n_joints*3)
            x = tf.reshape(sample, shape)
            for i, n_h in enumerate(n_c):
                x = tf.layers.dense(
                    x, n_h, activation=_activations[self.params['activation']],
                    kernel_initializer=initializer,
                    name='dense%d' % i)
            x = tf.layers.dense(
                x, 1, activation=None, kernel_initializer=initializer,
                name='dense_final')
            x = tf.squeeze(x, 1)
        return x

    def get_random_generator_input(self):
        """Get a generator sample suitable for training."""
        return tf.random_normal(
            (self.params['batch_size'], self.params['n_z']),
            dtype=tf.float32, name='z')

    def _get_np_data(self):
        """Get numpy data used to train critic."""
        dataset_params = self.params['dataset']
        dataset = get_dataset(dataset_params['type'])
        dataset = filter_dataset(
            dataset, modes=['train'], **dataset_params['filter_kwargs'])
        dataset = copy_group(dataset)
        dataset = normalize_dataset(
            dataset, **dataset_params['normalize_kwargs'])
        p3w = dataset_to_p3w(dataset, rotate_front=True, recenter_xy=True)
        return p3w

    def get_real_sample(self):
        """Get a tensor of real human poses for use by critic."""
        p3_np = self._get_np_data()
        dataset = tf.data.Dataset.from_tensor_slices(p3_np)
        dataset = dataset.shuffle(100000).repeat().batch(
            self.params['batch_size'])
        p3 = dataset.make_one_shot_iterator().get_next()
        return p3

    def gan_estimator(self, config=None):
        """Get a GANEstimator for this gan."""
        loss_type = self.params['loss_type']
        generator_loss_fn = _generator_losses[loss_type]
        discriminator_loss_fn = _discriminator_losses[loss_type]

        # generator_optimizer = tf.train.AdamOptimizer(0.1, 0.5)
        # discriminator_optimizer = tf.train.AdamOptimizer(0.1, 0.5)
        generator_optimizer = tf.train.RMSPropOptimizer(1e-4)
        discriminator_optimizer = tf.train.RMSPropOptimizer(1e-4)
        # generator_optimizer = tf.train.AdamOptimizer(1e-4)
        # discriminator_optimizer = tf.train.AdamOptimizer(1e-4)
        if 'discriminator_clip_val' in self.params:
            clip_val = self.params['discriminator_clip_val']
            original_fn = discriminator_optimizer.apply_gradients

            def modified_fn(*args, **kwargs):
                op = original_fn(*args, **kwargs)
                var_list = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='Discriminator')
                with tf.control_dependencies([op]):
                    ops = [tf.assign(v, tf.clip_by_value(
                                     v, -clip_val, clip_val))
                           for v in var_list]
                    op = tf.group(*ops)
                return op
            discriminator_optimizer.apply_gradients = modified_fn

        return tfgan.estimator.GANEstimator(
            model_dir=self.model_dir,
            generator_fn=self.get_generator_sample,
            discriminator_fn=self.get_critic_logits,
            generator_loss_fn=generator_loss_fn,
            discriminator_loss_fn=discriminator_loss_fn,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer
        )

    @property
    def model_dir(self):
        """Get the model directoy in which this builder saves."""
        return gan_model_dir(self.id)

    @property
    def latest_checkpoint(self):
        """Get the latest checkpoint for this model."""
        return tf.train.latest_checkpoint(self.model_dir)
