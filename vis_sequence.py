import numpy as np
from human_pose_util.register import register_defaults, dataset_register, \
    skeleton_register
from gan import load_params
from generate_inferences import results_path
import h5py

model_name = 'big'
dataset_id = 'h3m_consistent_scaled_c1_10fps'
example_index = 0

register_defaults()
params = load_params(model_name)
dataset = dataset_register[dataset_id]['eval']
skeleton = skeleton_register[dataset.attrs['skeleton_id']]

with h5py.File(results_path, 'r') as f:
    group = f[model_name][dataset_id]
    key = list(dataset.keys())[example_index]
    ground_truth = np.array(dataset[key]['p3w'])
    inferred = np.array(group[key]['p3w'])


def vis(ground_truth, inferred):
    # from pose_vis.animated_scene import add_limb_collection_animator
    # from pose_vis.animated_scene import run
    # fps = dataset.attrs['fps']
    # add_limb_collection_animator(skeleton, inferred, fps, linewidth=2.0)
    # add_limb_collection_animator(skeleton, ground_truth, fps, linewidth=4.0)
    # run(fps=10)
    from human_pose_util.skeleton import vis3d
    import matplotlib.pyplot as plt
    for i in range(0, len(ground_truth), 10):
        ax = vis3d(skeleton, ground_truth[i], linewidth=4.0)
        vis3d(skeleton, inferred[i], ax=ax)
        plt.show()


vis(ground_truth, inferred)
