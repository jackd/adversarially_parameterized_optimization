"""Script for reporting procrustes errors."""
import h5py
from human_pose_util.dataset.h3m.report import report, sequence_report
from adversarially_parameterized_optimization.serialization import results_path


def inference_report(inference_id, use_s14=False):
    """
    Print procruste errors for previously inferred sequences.

    If use_s14, only the 14 joints in s14 are considered in the averages
    (procruste alignment is still done on the entire skeleton).
    """
    with h5py.File(results_path, 'a') as results:
        results = results[inference_id]
        print('Individual proc_err')
        report(results, use_s14=use_s14)
        print('----------------')
        print('----------------')
        print('Sequence proc_err')
        sequence_report(results, use_s14=use_s14)


if __name__ == '__main__':
    import argparse
    from human_pose_util.register import register_defaults
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'inference_id',
        help='id of inference spec defined in inference_params')
    parser.add_argument('--use_s14', action='store_true')
    args = parser.parse_args()
    register_defaults()
    inference_report(args.inference_id, args.use_s14)
