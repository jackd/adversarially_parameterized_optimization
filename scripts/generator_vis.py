"""Script for visualizing generator samples."""
import tensorflow as tf
import matplotlib.pyplot as plt
from adversarially_parameterized_optimization.gan import \
    get_random_generator_input, get_generator_sample
from adversarially_parameterized_optimization.serialization import \
    load_gan_params, gan_model_dir
from human_pose_util.register import skeleton_register, dataset_register

from human_pose_util.skeleton import vis3d


def vis(gan_id):
    params = load_gan_params(gan_id)
    skeleton = skeleton_register[
        dataset_register[params['dataset']]['train'].attrs['skeleton_id']]

    print('Building graph...')
    graph = tf.Graph()
    with graph.as_default():
        gen_input = get_random_generator_input(params)
        sample = get_generator_sample(gen_input, params, reuse=False)
        generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    print('Starting session...')
    with tf.Session(graph=graph) as sess:
        print('Restoring variables...')
        saver = tf.train.Saver(var_list=generator_vars)
        saver.restore(
            sess, tf.train.latest_checkpoint(gan_model_dir(gan_id)))
        print('Generating...')
        sample_data = sess.run(sample)

    print('Visualizing...')
    for s in sample_data:
        vis3d(skeleton, s)
        plt.show()


if __name__ == '__main__':
    import argparse
    from human_pose_util.register import register_defaults
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'gan_id',
        help='id of GAN spec defined in gan_params')
    args = parser.parse_args()
    register_defaults()
    vis(args.gan_id)
