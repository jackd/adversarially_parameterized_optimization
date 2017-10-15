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

# Algorithm Overview
The premise of the paper is to train a GAN to simultaneously learn a parameterization of the feasible human pose space along with a feasibility loss function.

During inference, a standard off-the-shelf optimizer infers all poses from sequence almost-independently (the scale is shared between frames, which has no effect on the results (since errors are on the procruste-aligned inferences which optimize over scale) but makes the visualizations easier to interpret).

# Repository Structure
Each GAN is identified by a `gan_id`. Hyperparameters defining the network structures and datasets from which they should be trained are specified in `gan_params/gan_id.json`. A couple (those with results highlighted in the paper) are provided, `h3m_big`, `h3m_small` and `eva_big`. Note that compared to typical neural networks, these are still tiny, so the difference in size should result in a negligible difference in training/inference time.

Similarly, each inference run is identified by an `inference_id`, the parameters of which are defined in `inference_params/inference_id.json`.

- `gan_tf`: Provides classes for GANs used.
- `human_pose_util`: Provides utility functions for 3D human pose estimation including geometric transforms, visualizations and dataset reading
- `gan`: provides application-specific GANs based on specifications in `gan_params`
- `serialization.py`: i/o related functions for loading hyper-parameters/results

Scripts:
- `train.py`: Trains a GAN specified by a `json` file in `gan_params`
- `gan_generator_vis.py`: visualization script for a trained GAN generator
- `interactive_gan_generator_vis.ipynb`: interactive jupyter/ipython notebook for visualizing a trained GAN generator
- `generate_inferences.py`: Generates inferences based on parameters specified by a `json` file in `inference_params`
- `h3m_report.py`/`eva_report.py`: reporting scripts for generated inferences.
- `vis_sequecne.py`: visualization script for entire inferred sequence.


# Usage
  1. Clone this repository and add the location and the cloned directory to your `PYTHON_PATH`
  ```
  cd path/to/parent_folder
  git clone https://github.com/jackd/adversarially_parameterized_optimization.git
  export PYTHONPATH=/path/to/parent_folder:/path/to/parent_folder/adversarially_parameterized_optimization:$PYTHONPATH
  cd adversarially_parameterized_optimization
  ```
  2. Define a GAN model by creating a `gan_params/gan_id.json` file, or select one of the existing ones.
  3. Setup the specified dataset or create your own. See below for details on how to use `h3m`/`eva` datasets supplied.
  4. Train the GAN
  ```
  python train.py gan_id --max_steps=1e7
  ```
  Our experiments were conducted on an NVidia K620 Quadro GPU with 2GB memory. Training runs at ~600 batches per second with a batch size of 128. For 10 million steps (likely excessive) this takes around 4.5 hours.

  View training progress and compare different runs using tensorboard:
  ```
  tensorboard --logdir=models
  ```
  5. (Optional) Check your generator is behaving well by running `gan_generator_vis.py model_id` or interactively by running `interactive_gan_generator_vis.ipynb` and modifying the `model_id`.
  6. Define an inference specification by creating an `inference_params/inference_id.json` file, or select one of the defaults provided.
  7. Generate inference
  ```
  python generate_inferences.py inference_id
  ```
  Sequence optimization runs at ~5-10fps (speed-up compared to 1fps reported in paper due to reimplementation efficiencies rather than different ideas).

  This will save results in `results.hdf5` in the `inference_id` group.
  8. See the results!
    * `h3m_report.py` or `eva_report.py` depending on the dataset gives qualitative results
  ```
  python report.py eval_id
  ```
    * `vis_sequence.py` visualizes inferences
  Note that results are quite unstable with respect to GAN training. You may get considerably different quantitative results than those published in the paper, though qualitative behaviour should be similar.

# Serialization
To aid with experiments with different parameter sets, model/inference parameters are saved in `json` for ease of parsing and human readability. To allow for extensibility, `human_pose_util` maintains registers for different datasets and skeletons.

The scripts in this project register some default h3m/eva datasets using `register_defaults`. While normally fast, some data conversion is performed the first time this function is run and requires the original datasets be available with paths defined (see below). If you only wish to experiment with one dataset -- e.g. `h3m` -- modify `register_default_datasets` and `register_default_skeletons` in `human_pose_util.register` by removing registration lines related to `eva`.

If you implement your own datasets/skeletons, either add their registrations to the default functions, or edit the relevant scripts to register them manually.

# Datasets
`human_pose_util` comes with support for [Human3.6M](http://vision.imar.ro/human3.6m/description.php) (h3m) and [HumanEva_I](http://humaneva.is.tue.mpg.de/datasets_human_1) (eva) datasets. Due to licensing issues these are not provided here - see the respective websites for details.

## Setting up datasets

### [Human3.6M](http://vision.imar.ro/human3.6m/description.php) (h3m)
To work with the Human3.6M dataset, you must have the relevant `.cdf` files in an uncompressed local directory, referenced here as `MY_H3M_DIRECTORY`. For licensing reasons, we cannot provide the raw Human3.6m data. Please consult the [website](http://vision.imar.ro/human3.6m/description.php) to source the original data. This directory must have the following structure:
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

### [HumanEva_I](http://humaneva.is.tue.mpg.de/datasets_human_1) (eva)
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

If your dataset uses a different skeleton from those provided (see `human_pose_util.skeleton.Skeleton`), you'll need to precede this with a similar skeleton registration line
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
For the sake of reproducibility, this branch is now frozen aside from the `Known Bugs` section below which will be updated as they become apparent. Development will continue on other branches.

# Known Bugs
None at this time.
