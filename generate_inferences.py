"""Provides inference functions.."""
import os
import numpy as np
import tensorflow as tf
import h5py
from time import time
from gan import load_params, model_dir, get_generator_sample
from human_pose_util.register import dataset_register
from human_pose_util.transforms.tf_impl import tf_impl

_root_dir = os.path.realpath(os.path.dirname(__file__))
results_path = os.path.join(_root_dir, 'results.hdf5')


def infer_sequence_poses(model_id, p2, r, t, f, c):
    """Get 3d pose inference in world coordinates for a sequence."""
    # p2, r, t, f, c = (example[k] for k in ['p3c', 'r', 't', 'f', 'c'])
    params = load_params(model_id)
    n_z = params['n_z']
    n_frames = len(p2)

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
            # np.ones((n_frames,), dtype=np.float32)*5,
            np.zeros((n_frames,), dtype=np.float32),
            dtype=tf.float32, name='y0')

        opt_vars = [z, scale, phi, x0, y0]

        normalized_p3 = get_generator_sample(z, params, reuse=False)
        p3w = tf_impl.rotate_about(
            normalized_p3*scale, tf.expand_dims(phi, axis=-1), dim=2)
        offset = tf.stack([x0, y0, tf.zeros_like(x0)], axis=-1)
        p3w = p3w + tf.expand_dims(offset, axis=-2)
        # p3c = tf.einsum('ijk,kl->ijl', normalized_p3, Rt) + t
        p3c = tf_impl.transform_frame(p3w, r, t)
        p2i = tf_impl.project(p3c, f=f, c=c)
        loss = 1000*tf.nn.l2_loss(p2i - p2)

    with tf.Session(graph=graph) as sess:
        opt_vars_set = set(opt_vars)
        model_vars = [
            v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if v not in opt_vars_set]
        saver = tf.train.Saver(model_vars)
        saver.restore(
            sess, tf.train.latest_checkpoint(model_dir(model_id)))
        sess.run(tf.variables_initializer(opt_vars))
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, opt_vars)
        t = time()
        optimizer.minimize(sess)
        dt = time() - t
        p3w_vals = sess.run(p3w)

    return p3w_vals, dt


def generate_all(model_id, dataset_id, overwrite=False):
    """Generate all results for the specified model/dataset."""
    with h5py.File(results_path, 'a') as f:
        group = f.require_group('%s/%s' % (model_id, dataset_id))
        group.attrs['model_id'] = model_id
        group.attrs['dataset_id'] = dataset_id
        dataset = dataset_register[dataset_id]['eval']
        n_examples = len(dataset)
        for i, key in enumerate(dataset):
            print('Processing sequence %d / %d' % (i + 1, n_examples))
            print(key)
            ex_group = group.require_group(key)
            if 'p3w' in ex_group:
                if overwrite:
                    del ex_group['p3w']
                else:
                    continue
            example = dataset[key]
            p2 = example['p2']
            r, t, f, c = (example.attrs[k] for k in ['r', 't', 'f', 'c'])
            p3w, dt = infer_sequence_poses(model_id, p2, r, t, f, c)
            n_frames = len(p2)
            fps = n_frames / dt
            print('Completed %d frames in %.2fs @ %.2f fps'
                  % (n_frames, dt, fps))
            ex_group.create_dataset('p3w', data=p3w)
            ex_group.attrs['inference_time'] = dt
            ex_group.attrs['inference_fps'] = fps


if __name__ == '__main__':
    from human_pose_util.register import register_defaults
    register_defaults()
    model_id = 'big'
    dataset_id = 'h3m_consistent_scaled_c1_10fps'
    overwrite = True
    generate_all(model_id, dataset_id, overwrite)
