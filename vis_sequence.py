"""Script for visualizing an inferred sequence along with ground truth data."""
import random
import numpy as np
import h5py
from serialization import results_path, load_inference_params
from human_pose_util.register import get_dataset, get_skeleton


def vis_data_glumpy(skeleton, fps, ground_truth, inferred):
    """Visualizes ground_truth/inferred data using glumpy."""
    from human_pose_util.animation.animated_scene import \
        add_limb_collection_animator
    from human_pose_util.animation.animated_scene import run
    add_limb_collection_animator(skeleton, inferred, fps, linewidth=2.0)
    add_limb_collection_animator(skeleton, ground_truth, fps, linewidth=4.0)
    run(fps=fps)


def vis_data_plt(skeleton, ground_truth, inferred):
    """Visualizes ground_truth/inferred data using matplotlib.pyplot."""
    from human_pose_util.skeleton import vis3d
    import matplotlib.pyplot as plt
    for i in range(0, len(ground_truth), 10):
        ax = vis3d(skeleton, ground_truth[i], linewidth=4.0)
        vis3d(skeleton, inferred[i], ax=ax)
        plt.show()


def vis_sequence(inference_id, example_id=None, use_plt=True):
    """
    Visualize a previously inferred sequence.

    Args:
        inference_id: id of the relevant inference set.
        example_id: example id to visualize. If none, a random example is
            selected
        use_plt: if true, uses matplotlib.pyplot for visualizations, otherwise
            uses glumpy. glumpy visualization is an animation, while plt
            version gives a number of frames.
    """
    inference_params = load_inference_params(inference_id)
    dataset = get_dataset(inference_params['dataset']['type'])
    skeleton = get_skeleton(dataset.attrs['skeleton_id'])

    with h5py.File(results_path, 'r') as f:
        group = f[inference_id]
        if example_id is None:
            example_id = random.sample(list(group.keys()), 1)[0]
        example = dataset[example_id]
        fps = example.attrs['fps']
        ground_truth = np.array(example['p3w'])
        inferred = np.array(group[example_id]['p3w'])

    if use_plt:
        vis_data_plt(skeleton, ground_truth, inferred)
    else:
        vis_data_glumpy(skeleton, fps, ground_truth, inferred)


if __name__ == '__main__':
    import argparse
    from serialization import register_defaults

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'inference_id',
        help='id of inference spec defined in inference_params')
    parser.add_argument(
        '-e', '--example_id', type=str, default=None,
        help='example_id to visualize')
    parser.add_argument('-p', '--use_plt', action="store_true")
    args = parser.parse_args()

    register_defaults()
    vis_sequence(args.inference_id, args.example_id, args.use_plt)
