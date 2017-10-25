from human_pose_util.dataset.normalize import normalize_dataset, filter_dataset
from human_pose_util.dataset.group import copy_group
from human_pose_util.register import get_dataset


def get_normalized_dataset(dataset_params):
    dataset = get_dataset(dataset_params['type'])
    dataset = filter_dataset(
        dataset, modes=['eval'], **dataset_params['filter_kwargs'])
    dataset = copy_group(dataset)
    dataset = normalize_dataset(dataset, **dataset_params['normalize_kwargs'])
    return dataset
