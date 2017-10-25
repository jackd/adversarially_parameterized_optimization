"""Script for visualizing generator samples."""
import tensorflow as tf
import matplotlib.pyplot as plt
from gan import GanBuilder
from human_pose_util.register import get_skeleton

from human_pose_util.skeleton import vis3d


def vis(gan_id):
    """Visualize output from the given gan."""
    builder = GanBuilder(gan_id)
    skeleton = get_skeleton(
        builder.params['dataset']['normalize_kwargs']['skeleton_id'])

    print('Building graph...')
    graph = tf.Graph()
    with graph.as_default():
        gen_input = builder.get_random_generator_input()
        with tf.variable_scope('Generator'):
            sample = builder.get_generator_sample(gen_input)
        generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    print('Starting session...')
    with tf.Session(graph=graph) as sess:
        print('Restoring variables...')
        saver = tf.train.Saver(var_list=generator_vars)
        saver.restore(
            sess, builder.latest_checkpoint)
        print('Generating...')
        sample_data = sess.run(sample)

    print('Visualizing...')
    for s in sample_data:
        vis3d(skeleton, s)
        plt.show()


if __name__ == '__main__':
    import argparse
    from serialization import register_defaults
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'gan_id',
        help='id of GAN spec defined in gan_params')
    args = parser.parse_args()
    register_defaults()
    vis(args.gan_id)
