"""Provides inference functions.."""
import numpy as np
import tensorflow as tf
import h5py
from time import time
from gan import GanBuilder
from serialization import load_inference_params, results_path
from serialization import inference_params_path
from human_pose_util.register import get_dataset, get_converter
from human_pose_util.dataset.normalize import normalize_dataset, filter_dataset
from human_pose_util.dataset.group import copy_group
from human_pose_util.transforms.tf_impl import tf_impl


def infer_sequence_poses(
        gan_id, p2, r, t, f, c, dt, loss_weights, tol,
        target_skeleton_id=None):
    """Get 3d pose inference in world coordinates for a sequence."""
    n_frames = len(p2)
    builder = GanBuilder(gan_id)
    skeleton_id = builder.params['dataset']['kwargs']['skeleton_id']
    if skeleton_id == target_skeleton_id:
        convert = None
    else:
        convert = get_converter(skeleton_id, target_skeleton_id).convert_tf

    graph = tf.Graph()
    with graph.as_default():

        z = tf.Variable(
             np.zeros((n_frames, builder.params['n_z']), dtype=np.float32),
             dtype=tf.float32, name='z')
        scale = tf.Variable(1.65, dtype=tf.float32, name='scale')
        phi = tf.Variable(
            np.zeros((n_frames,), dtype=np.float32),
            dtype=tf.float32, name='phi')
        x0 = tf.Variable(
            np.zeros((n_frames,), dtype=np.float32),
            dtype=tf.float32, name='x0')
        y0 = tf.Variable(
            np.zeros((n_frames,), dtype=np.float32),
            dtype=tf.float32, name='y0')

        opt_vars = [z, scale, phi, x0, y0]

        with tf.variable_scope('Generator'):
            normalized_p3 = builder.get_generator_sample(z)
        with tf.variable_scope('Discriminator'):
            critic_logits = builder.get_critic_logits(normalized_p3, z)
        if convert is not None:
            normalized_p3 = convert(normalized_p3)
        p3w = tf_impl.rotate_about(
            normalized_p3*scale, tf.expand_dims(phi, axis=-1), dim=2)
        offset = tf.stack([x0, y0, tf.zeros_like(x0)], axis=-1)
        p3w = p3w + tf.expand_dims(offset, axis=-2)
        p3c = tf_impl.transform_frame(p3w, r, t)
        p2i = tf_impl.project(p3c, f=f, c=c)

        losses = {}
        losses['consistency'] = tf.nn.l2_loss(p2i - p2)
        losses['critic'] = -tf.reduce_sum(critic_logits)
        vel = (p3w[1:] - p3w[:-1]) / dt
        losses['smoothness'] = tf.nn.l2_loss(vel)
        speed2 = tf.reduce_sum(vel**2, axis=2)
        losses['glide'] = tf.reduce_sum(tf.reduce_min(speed2, axis=1))
        loss_terms = [
            losses[k[:-7]]*v for k, v in loss_weights.items() if v > 0]
        if len(loss_terms) == 0:
            raise Exception('At least one of loss_weights must be positive')
        elif len(loss_terms) == 1:
            loss = loss_terms[0]
        else:
            loss = tf.add_n(loss_terms, name='combined_loss')

    with tf.Session(graph=graph) as sess:
        opt_vars_set = set(opt_vars)
        model_vars = [
            v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if v not in opt_vars_set]
        saver = tf.train.Saver(model_vars)
        saver.restore(sess, builder.latest_checkpoint)
        sess.run(tf.variables_initializer(opt_vars))
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss, opt_vars, tol=tol)
        t = time()
        optimizer.minimize(sess)
        dt = time() - t
        p3w_vals = sess.run(p3w)

    return p3w_vals, dt


def generate_all(inference_id, overwrite=False):
    """Generate all results for the specified model/dataset."""
    inference_params = load_inference_params(inference_id)
    gan_id = inference_params['gan_id']
    dataset_params = inference_params['dataset']
    dataset = get_dataset(dataset_params['type'])
    dataset = filter_dataset(
        dataset, modes=['eval'], **dataset_params['filter_kwargs'])
    dataset = copy_group(dataset)
    dataset = normalize_dataset(dataset, **dataset_params['normalize_kwargs'])
    target_skeleton_id = dataset.attrs['skeleton_id']
    loss_weights = {k: inference_params[k] for k in [
        'critic_weight',
        'smoothness_weight',
        'glide_weight',
        'consistency_weight',
    ]}
    tol = inference_params['tol']
    with h5py.File(results_path, 'a') as f:
        group = f.require_group(inference_id)
        group.attrs['params_path'] = inference_params_path(inference_id)
        n_examples = len(dataset)
        for i, key in enumerate(dataset):
            print('Processing sequence %d / %d' % (i + 1, n_examples))
            ex_group = group.require_group(key)
            if 'p3w' in ex_group and not overwrite:
                continue
            sequence = dataset[key]
            dt = 1./sequence.attrs['fps']
            p2 = sequence['p2']
            r, t, f, c = (sequence.attrs[k] for k in ['r', 't', 'f', 'c'])
            p3w, dt = infer_sequence_poses(
                gan_id, p2, r, t, f, c, dt, loss_weights, tol,
                target_skeleton_id)
            n_frames = len(p2)
            fps = n_frames / dt
            print('Completed %d frames in %.2fs @ %.2f fps'
                  % (n_frames, dt, fps))
            if 'p3w' in ex_group:
                del ex_group['p3w']
            ex_group.create_dataset('p3w', data=p3w)
            ex_group.attrs['inference_time'] = dt
            ex_group.attrs['inference_fps'] = fps


if __name__ == '__main__':
    import argparse
    from serialization import register_defaults
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'inference_id',
        help='id of inference spec defined in inference_params')
    parser.add_argument(
        '-o', '--overwrite', action='store_true',
        help='overwrite data if present')
    args = parser.parse_args()
    register_defaults()
    generate_all(args.inference_id, args.overwrite)
