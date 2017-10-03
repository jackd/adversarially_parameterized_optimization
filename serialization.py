"""Provides functions for loading parameters and getting paths."""
import os
import json

_root_dir = os.path.realpath(os.path.dirname(__file__))


def gan_params_path(gan_id):
    """Get path to model parameter file."""
    return os.path.join(_root_dir, 'gan_params', '%s.json' % gan_id)


def gan_model_dir(gan_id):
    """Get the directory used by the GAN to save."""
    return os.path.join(_root_dir, 'models', gan_id)


def _checked_load(path):
    if not os.path.isfile(path):
        raise ValueError('No parameter file at %s' % path)
    with open(path, 'r') as f:
        params = json.load(f)
    return params


def load_gan_params(gan_id):
    """Load model parameters for the specified model."""
    return _checked_load(gan_params_path(gan_id))


def inference_params_path(inference_id):
    """Get the path to the specified inference parameters file."""
    return os.path.join(
        _root_dir, 'inference_params', '%s.json' % inference_id)


def load_inference_params(inference_id):
    """Load inference parameters with the specified id."""
    return _checked_load(inference_params_path(inference_id))


results_path = os.path.join(_root_dir, 'results.hdf5')
