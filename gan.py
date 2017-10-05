"""Provides GAN for parameterization/feasibility loss learning."""
from __future__ import division
import numpy as np
import tensorflow as tf
from gan_tf.w_gan import \
    WeightClippedWGan, GradientPenalizedWGan
from human_pose_util.register import skeleton_register, dataset_register
from human_pose_util.skeleton import front_angle
from human_pose_util.transforms.np_impl import rotate_about
from adversarially_parameterized_optimization.serialization import \
    load_gan_params, gan_model_dir


_activations = {
    'relu': tf.nn.relu,
    'elu': tf.nn.elu
}


def _get_n_joints(params):
    dataset = dataset_register[params['dataset']]['train']
    skeleton = skeleton_register[dataset.attrs['skeleton_id']]
    n_joints = skeleton.n_joints
    return n_joints


def get_generator_sample(features, params, reuse):
    """Get the generator sample."""
    n_g = params['n_g']
    n_joints = _get_n_joints(params)
    x = features
    with tf.variable_scope('generator', reuse=reuse):
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        for i, n_h in enumerate(n_g):
            x = tf.layers.dense(
                x, n_h, activation=_activations[params['activation']],
                kernel_initializer=initializer, reuse=reuse,
                name='dense%d' % i)
        x = tf.layers.dense(
            x, n_joints * 3, activation=None, kernel_initializer=initializer,
            reuse=reuse, name='dense_final')
        shape = x.shape.as_list()[:-1] + [n_joints, 3]
        for i, s in enumerate(shape):
            if s is None:
                shape[i] = -1
                break
        x = tf.reshape(x, shape)
    return x


def get_critic_logits(sample, params, reuse):
    """Critic logits function used in GAN."""
    n_joints = _get_n_joints(params)
    n_c = params['n_c']
    with tf.variable_scope('critic', reuse=reuse):
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
                x, n_h, activation=_activations[params['activation']],
                kernel_initializer=initializer, reuse=reuse,
                name='dense%d' % i)
        x = tf.layers.dense(
            x, 1, activation=None, kernel_initializer=initializer, reuse=reuse,
            name='dense_final')
        x = tf.squeeze(x, 1)
    return x


def get_random_generator_input(params):
    """Get a generator sample suitable for training."""
    batch_size = params['batch_size']
    n_z = params['n_z']
    return tf.random_normal(
        (batch_size, n_z), dtype=tf.float32, name='z')


def _get_np_data(params):
    """Get numpy data used to train critic."""
    from human_pose_util.dataset.spec import calculate_heights
    dataset = dataset_register[params['dataset']]['train']
    heights = calculate_heights(dataset)
    p3s = []
    for key in dataset:
        example = dataset[key]
        height = heights[example.attrs['subject_id']]
        p3 = np.array(dataset[key]['p3w'], dtype=np.float32)
        p3 /= height
        p3s.append(p3)
    skeleton = skeleton_register[dataset.attrs['skeleton_id']]
    p3s = np.concatenate(p3s, axis=0)
    # hips above origin
    p3s[..., :2] -= p3s[:, 0:1, :2]
    # rotate hips to front
    phi = front_angle(p3s, skeleton)
    p3s = rotate_about(p3s, -np.expand_dims(phi, axis=1), dim=2)
    return p3s


def get_real_sample(params):
    """Get a tensor of real human poses for use by critic."""
    p3_np = _get_np_data(params)
    dataset = tf.contrib.data.Dataset.from_tensor_slices(p3_np)
    dataset = dataset.shuffle(100000).repeat().batch(params['batch_size'])
    p3 = dataset.make_one_shot_iterator().get_next()
    return p3


def pose_gan(gan_id, config=None):
    """Get the named GAN. See `params_path` to view parameters."""
    params = load_gan_params(gan_id)
    wgan_type = params['wgan_type']
    args = [get_generator_sample, get_critic_logits]
    kwargs = dict(
        config=config, params=params, model_dir=gan_model_dir(gan_id),
        name=gan_id)
    if wgan_type == 'weight_clipped':
        return WeightClippedWGan(*args, **kwargs)
    elif wgan_type == 'gradient_penalized':
        return GradientPenalizedWGan(*args, **kwargs)
    else:
        raise ValueError(
            'Invalid wgan_type %s in params. Should be one of '
            '[weight_clipped, gradient_penalized]' % wgan_type)
