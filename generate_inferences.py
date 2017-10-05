"""Provides inference functions.."""
import numpy as np
import tensorflow as tf
import h5py
from time import time
from gan import get_generator_sample, get_critic_logits
from serialization import \
    load_inference_params, load_gan_params, gan_model_dir, results_path
from human_pose_util.register import dataset_register, skeleton_register
from human_pose_util.skeleton import s24_to_s14_converter
from human_pose_util.dataset.eva.skeleton import s16_to_s14_converter
from human_pose_util.transforms.tf_impl import tf_impl
from human_pose_util.dataset.h3m.skeleton import s24
from human_pose_util.dataset.eva.skeleton import s16, s14


def infer_sequence_poses(
        gan_id, p2, r, t, f, c, dt, loss_weights, tol, target_skeleton=None):
    """Get 3d pose inference in world coordinates for a sequence."""
    # p2, r, t, f, c = (example[k] for k in ['p3c', 'r', 't', 'f', 'c'])
    params = load_gan_params(gan_id)
    n_z = params['n_z']
    n_frames = len(p2)

    if target_skeleton is None:
        convert = None
    else:
        train_dataset = dataset_register[params['dataset']]['train']
        output_skeleton = skeleton_register[train_dataset.attrs['skeleton_id']]
        if output_skeleton == s24:
            if target_skeleton == s14:
                convert = s24_to_s14_converter().convert_tf
            elif target_skeleton == s24:
                convert = None
            else:
                raise ValueError('Not valid skeleton combination')
        elif output_skeleton == s16:
            if target_skeleton == s14:
                convert = s16_to_s14_converter().convert_tf
            elif target_skeleton == s16:
                convert = None
            else:
                raise ValueError('Not valid skeleton combination')
        elif output_skeleton == s14:
            convert = None
        else:
            raise ValueError('Not valid skeleton combination')

    graph = tf.Graph()
    with graph.as_default():

        z = tf.Variable(
             np.zeros((n_frames, n_z), dtype=np.float32),
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

        normalized_p3 = get_generator_sample(z, params, reuse=False)
        critic_logits = get_critic_logits(normalized_p3, params, False)
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
        saver.restore(
            sess, tf.train.latest_checkpoint(gan_model_dir(gan_id)))
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
    dataset_id = inference_params['dataset']
    dataset = dataset_register[dataset_id]['eval']
    loss_weights = {k: inference_params[k] for k in [
        'critic_weight',
        'smoothness_weight',
        'glide_weight',
        'consistency_weight',
    ]}
    tol = inference_params['tol']
    with h5py.File(results_path, 'a') as f:
        group = f.require_group(inference_id)
        for k, v in inference_params.items():
            if k not in group.attrs:
                group.attrs[k] = v
        n_examples = len(dataset)
        for i, key in enumerate(dataset):
            print('Processing sequence %d / %d' % (i + 1, n_examples))
            print(key)
            ex_group = group.require_group(key)

            if 'p3w' in ex_group and not overwrite:
                continue
            example = dataset[key]
            dt = 1./example.attrs['fps']
            p2 = example['p2']
            r, t, f, c = (example.attrs[k] for k in ['r', 't', 'f', 'c'])
            target_skeleton = skeleton_register[dataset.attrs['skeleton_id']]
            p3w, dt = infer_sequence_poses(
                gan_id, p2, r, t, f, c, dt, loss_weights, tol, target_skeleton)
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
    from human_pose_util.register import register_defaults
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
