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

Code used to generate results for the paper has been frozen and can be found in the `3dv2017` branch. Bug fixes and extensions will be applied to other branches.

# Algorithm Overview
The premise of the paper is to train a GAN to simultaneously learn a parameterization of the feasible human pose space along with a feasibility loss function.

During inference, a standard off-the-shelf optimizer infers all poses from sequence almost-independently (the scale is shared between frames, which has no effect on the results (since errors are on the procruste-aligned inferences which optimize over scale) but makes the visualizations easier to interpret).

# Repository Structure
Each GAN is identified by a `gan_id`. Hyperparameters defining the network structures and datasets from which they should be trained are specified in `gan_params/gan_id.json`. A couple (those with results highlighted in the paper) are provided, `h3m_big`, `h3m_small` and `eva_big`. Note that compared to typical neural networks, these are still tiny, so the difference in size should result in a negligible difference in training/inference time.

Similarly, each inference run is identified by an `inference_id`, the parameters of which are defined in `inference_params/inference_id.json`.
 including geometric transforms, visualizations and dataset reading
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
  1. Setup the external repositories:
    * [`human_pose_util`](https://github.com/jackd/human_pose_util)
  2. Clone this repository and add the location and the parent directory(s) to your `PYTHONPATH`
  ```
  cd path/to/parent_folder
  git clone https://github.com/jackd/adversarially_parameterized_optimization.git
  git clone https://github.com/jackd/human_pose_util.git
  export PYTHONPATH=/path/to/parent_folder:$PYTHONPATH
  cd adversarially_parameterized_optimization
  ```
  3. Define a GAN model by creating a `gan_params/gan_id.json` file, or select one of the existing ones.
  4. Setup the relevant dataset(s) or create your own as described in [`human_pose_util`](https://github.com/jackd/human_pose_util).
  5. Train the GAN
  ```
  python train.py gan_id --max_steps=1e7
  ```
  Our experiments were conducted on an NVidia K620 Quadro GPU with 2GB memory. Training runs at ~600 batches per second with a batch size of 128. For 10 million steps (likely excessive) this takes around 4.5 hours.

  View training progress and compare different runs using tensorboard:
  ```
  tensorboard --logdir=models
  ```
  6. (Optional) Check your generator is behaving well by running `gan_generator_vis.py model_id` or interactively by running `interactive_gan_generator_vis.ipynb` and modifying the `model_id`.
  7. Define an inference specification by creating an `inference_params/inference_id.json` file, or select one of the defaults provided.
  8. Generate inference
  ```
  python generate_inferences.py inference_id
  ```
  Sequence optimization runs at ~5-10fps (speed-up compared to 1fps reported in paper due to reimplementation efficiencies rather than different ideas).

  This will save results in `results.hdf5` in the `inference_id` group.
  9. See the results!
    * `h3m_report.py` or `eva_report.py` depending on the dataset gives qualitative results
  ```
  python report.py eval_id
  ```
    * `vis_sequence.py` visualizes inferences
  Note that results are quite unstable with respect to GAN training. You may get considerably different quantitative results than those published in the paper, though qualitative behaviour should be similar.

# Serialization
To aid with experiments with different parameter sets, model/inference parameters are saved in `json` for ease of parsing and human readability. To allow for extensibility, [`human_pose_util`](https://github.com/jackd/human_pose_util) maintains registers for different datasets and skeletons.

See the [README](https://github.com/jackd/human_pose_util/README.md) for details on setting up/preprocessing of datasets or implementing your own.

The scripts in this project register some default h3m/eva datasets using `register_defaults`. While normally fast, some data conversion is performed the first time this function is run for each dataset and requires the original datasets be available with paths defined (see below). If you only wish to experiment with one dataset -- e.g. `h3m` -- modify the default argument values for `register_defaults`, e.g. `def register_defaults(h3m=True, eva=False):` (or the relevant function calls).

If you implement your own datasets/skeletons, either add their registrations to the default functions, or edit the relevant scripts to register them manually.

# Datasets
See [`human_pose_util`](https://github.com/jackd/human_pose_util) repository for instructions for setting up datasets.

# Requirements
For training/inference:
- tensorflow 1.4
- numpy
- h5py
For visualizations:
- matplotlib
- glumpy (install from source may reduce issues)
For initial human 3.6m dataset transformations:
- spacepy (for initial human 3.6m dataset conversion to hdf5)

# Development
This branch will be actively maintained, updated and extended. For code used to generate results for the publication, see the `3dv2017` branch.

# Contact
Please report any issues/bugs. Feature requests in this repository will largely be ignored, but will be considered if made in independent repositories.

Email contact to discuss ideas/collaborations welcome: `dominic.jack@hdr.qut.edu.au`.
