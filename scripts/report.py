import h5py

from human_pose_util.dataset.h3m.report import report, sequence_report
from adversarially_parameterized_optimization.serialization import results_path


def inference_report(inference_id):
    with h5py.File(results_path, 'a') as results:
        results = results[inference_id]
        print('Individual proc_err')
        report(results)
        print('Sequence proc_err')
        sequence_report(results)


if __name__ == '__main__':
    import argparse
    from human_pose_util.register import register_defaults
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'inference_id',
        help='id of inference spec defined in inference_params')
    args = parser.parse_args()
    register_defaults()
    inference_report(args.inference_id)
