"""Script for reporting procrustes errors for h3m datasets."""
import h5py
from human_pose_util.dataset.h3m.report import proc_manager, \
    sequence_proc_manager
from serialization import results_path


def inference_report(
        inference_id, overwrite=False, use_s14=False):
    """
    Print procruste errors for previously inferred sequences.

    If use_s14, only the 14 joints in s14 are considered in the averages
    (procruste alignment is still done on the entire skeleton).
    """
    with h5py.File(results_path, 'a') as results:
        results = results[inference_id]
        print('Individual proc_err')
        proc_manager().report(results, overwrite=overwrite)
        print('----------------')
        print('Sequence proc_err')
        sequence_proc_manager().report(
            results, overwrite=overwrite)


if __name__ == '__main__':
    import argparse
    from serialization import register_defaults
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'inference_id',
        help='id of inference spec defined in inference_params')
    parser.add_argument(
        '-o', '--overwrite', action='store_true',
        help='overwrite existing data if present')
    args = parser.parse_args()
    register_defaults()
    inference_report(
        args.inference_id, overwrite=args.overwrite)
