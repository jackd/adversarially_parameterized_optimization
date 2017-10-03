Tensorflow implementation for 3DV 2017 conference paper "Adversarially Parameterized Optimization for 3D Human Pose Estimation".

```
@inproceedings{jack2017adversarially,
  title={Adversarially Parameterized Optimization for 3D Human Pose Estimation},
  author={Jack, Dominic and Maire, Frederic and Eriksson, Anders and Shirazi, Sareh},
  booktitle={3D Vision (3DV), 2017 Fifth International Conference on},
  year={2017},
  organization={IEEE}
}
```

This is currently being developed with several external (some private) repositories. I am in the process of ammending the code in those repositories such that they are suitable for public release. This is a temporary situation and should be resolved before the conference.

# Algorithm Overview
The premise of the paper is to train a GAN to simultaneously learn a parameterization of the feasible human pose space along with a feasibility loss function.

During inference, a standard off-the-shelf optimizer infers all poses from sequence almost-independently (the scale is shared between frames, which has no effect on the results (since errors are on the procruste-aligned inferences which optimize over scale) but makes the visualizations easier to interpret).

# Repository Structure
Each GAN is identified by a `gan_id`. Hyperparameters defining the network structures and datasets from which they should be trained are specified in `gan_params/gan_id.json`. A couple (those with results highlighted in the paper) are provided, `big` and `small`. Note that compared to typical neural networks, these are still tiny, so the difference in size should result in a negligible difference in training/inference time.

Similarly, each inference run is identified by an `eval_id`, the parameters of which are defined in `eval_params/eval_id.json`.

- `gan_tf`: (Currently in [external repository](https://github.com/jackd/gan_tf)) Provides classes for GANs used.
- `human_pose_util`: (Currently private) Provides utility functions for 3D human pose estimation including geometric transforms, visualizations and dataset reading
- `scripts`: contains training, inference and visualization scripts

# Usage
  1. Define a GAN model by creating a `gan_params/gan_id.json` file, or select one of `big`, `small`.
  2. Setup the specified dataset (if one of supplied h3m/eva) or create your own.
  3. Train the GAN
  ```
  python scripts/train.py gan_id --max_steps=1e7
  ```
  Our experiments were conducted on an NVidia K620 Quadro GPU with 2GB memory. Training runs at ~600 batches per second with a batch size of 128. For 10 million steps (likely excessive) this takes around 4.5 hours.
  4. (Optional) Check your generator is behaving well by running `scripts/generator_vis.py model_id` or running `scripts/interactive_generator_vis.ipynb` and modifying the `model_id`.
  5. Define an inference specification by creating an `inference_params/inference_id.json` file, or select one of the defaults provided.
  6. Run inference
  ```
  python scripts/generate_inferences.py inference_id
  ```
  Sequence optimization runs at ~5-10fps (speed-up compared to 1fps reported in paper due to reimplementation efficiencies rather than different ideas).

  This will save results in `results.hdf5` in the `inference_id` group.
  7. Check the results!
    * `scripts/report.py` gives qualitative results
  ```
  python scripts/report.py eval_id
  ```
    * `scripts/vis_sequence.py` visualizes inferences
  ```

# datasets
`human_pose_util` comes with support for Human3.6M and HumanEva_I datasets.

## Setting up datasets

### Human3.6M (h3m)
To work with the Human3.6M dataset, you must have the relevant `.cdf` files in an uncompressed local directory, referenced here as `MY_H3M_DIRECTORY`. This directory must have the following structure:
```
- MY_H3M_DIRECTORY
  - D2_positions
    - S1
      - Directions.54138969.cdf
      - ...
    - S5
      - ...
    ...
  - D3_positions
    - S1
    ...
  - D3_positions_mono
    - S1
    ...
  - Videos
    - S1
    ...
```

`Videos` aren't used in module, though the dataset has a `video_path` attribute which assumes the above structure.

To let the scripts know where to find the data, run the following in a terminal
```
export H3M_PATH=/path/to/MY_H3M_DIRECTORY
```

Consider adding this line to your `.bashrc` if you will be using this a lot.

### HumanEva_I (eva)
To work with the HumanEva_I dataset, you must have the uncompressed data available in `MY_EVA_1_DIR` which should have the following structure:
```
- MY_EVA_1_DIR
  - S1
    - Calibration_Data
      - BW1.cal
      ...
    - Image_Data
      - Box_1_(BW2).avi
      ...
    - Mocap_Data
      - Box_1.c3d
      - Box_1.mat
      ...
    - Sync_Data
      - Box_1_(BW1).ofs
      ...
  - S2
    ...
  ...
```

`Image_Data` is not used in this module, thought the dataset has a `video_path` attribute which assumes the above structure.

To let scripts know where to find the data, run the following in a terminal
```
export H3M_PATH=/home/jackd/Development/datasets/human3p6m/data
```

Consider adding this line to your `.bashrc` if you will be using this a lot.

## Registering a new dataset
A new dataset can be registered using
```
human_pose_util.register.dataset_register[dataset_id] = {
    'train': train_datastet,
    'eval': eval_dataset,
}
```

If your dataset uses a different skeleton (see `human_pose_util.Skeleton`), you'll need to precede this with a similar skeleton registration line
```
human_pose_util.register.skeleton_register[my_skeleton_id] = my_skeleton
```

After that, training/inference can procede as normal.

See `human_pose_util.dataset.h3m` and `human_pose_util.dataset.eva` for examples.

# Requirements
For training/inference:
- tensorflow 1.3
- numpy
- h5py
For visualizations:
- matplotlib
- glumpy (install from source may reduce issues)
For initial human 3.6m dataset transformations:
- spacepy (for initial human 3.6m dataset conversion to hdf5)

# Development
After the conference, this project will not be updated aside from bug fixes and documentation tweaks as reported. I will continue to work on constituent parts (e.g. [`gan_tf`](https://github.com/jackd/gan_tf), [`human_pose_util`]((https://github.com/jackd/human_pose_util))) in separate repositories, though the versions contained here will be frozen.

# TODO
- Eva Converters, test HumanEva_I dataset
- skeleton transforms - train/eval inconsistencies, e.g. training with s24, eval with s14. Use a `gan_builder`?
- openpose: code to generate heatmaps, inference from heatmaps
- fix README.md
  - human_pose_util
