import h5py
import numpy as np
from generate_inferences import results_path


def report(model_id, dataset_id):
    with h5py.File(results_path, 'r') as f:
        group = f[model_id][dataset_id]
        for k in group:
            example = group[k]
            print(k)
            print(np.mean(example['proc_err']))


if __name__ == '__main__':
    model_id = 'big'
    dataset_id = 'h3m_consistent_scaled_c1_10fps'
    report(model_id, dataset_id)
