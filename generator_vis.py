"""Script for visualizing generator samples."""
import tensorflow as tf
import matplotlib.pyplot as plt
from gan import get_random_generator_input, get_generator_sample
from gan import load_params, model_dir
from human_pose_util.register import skeleton_register, dataset_register
from human_pose_util.register import register_defaults
from human_pose_util.skeleton import vis3d

model_name = 'big'

register_defaults()
params = load_params(model_name)
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
    saver.restore(sess, tf.train.latest_checkpoint(model_dir(model_name)))
    print('Generating...')
    sample_data = sess.run(sample)

print('Visualizing...')
for s in sample_data:
    vis3d(skeleton, s)
    plt.show()
