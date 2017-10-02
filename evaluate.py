import h5py
from human_pose_util.register import dataset_register
from human_pose_util.evaluate import procrustes_error, \
    sequence_procrustes_error
from generate_inferences import results_path


def calc_procrustes_error(model_id, dataset_id, overwrite=False):
    with h5py.File(results_path, 'a') as group:
        addr = '%s/%s' % (model_id, dataset_id)
        if addr not in group:
            raise ValueError('No results for %s' % addr)
        group = group[addr]
        dataset = dataset_register[dataset_id]['eval']
        n_examples = len(dataset)
        for i, k in enumerate(dataset):
            print('Processing %d / %d' % (i+1, n_examples))
            print(k)
            inference_group = group[k]
            if 'proc_err' in inference_group and not overwrite:
                continue

            ground_truth = dataset[k]['p3w']
            inferred = inference_group['p3w']
            # print(ground_truth.shape)
            # print(inferred.shape)
            proc_err = procrustes_error(ground_truth, inferred)[0]
            # print(proc_err.shape)
            # exit()
            if 'proc_err' in inference_group:
                del inference_group['proc_err']
            inference_group.create_dataset('proc_err', data=proc_err)


if __name__ == '__main__':
    from human_pose_util.register import register_defaults
    register_defaults()
    model_id = 'big'
    dataset_id = 'h3m_consistent_scaled_c1_10fps'
    overwrite = True
    calc_procrustes_error(model_id, dataset_id, overwrite)
