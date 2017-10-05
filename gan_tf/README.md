# gan_tf
Classes similar to `tf.estimator.Estimator` for training generative adversarial networks (GANs) using tensorflow.

## Example usage
```
gan = Gan(generator_fn, critic_logits_fn, 'my_first_model_directory')

# Train for a million steps
gan.train(generator_input_fn, real_sample_fn, max_steps=10**6)
```
See `example/mnist/*` for MNIST training/generation example.

## Implemented GANs:
* `gan.py`: vanilla GAN
* `w_gan.py`: Wasserstein GAN

GAN logic based on [this work](https://github.com/wiseodd/generative-models/blob/master/GAN/).

Implementation and interface heavily based on [tf.estimator.Estimator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/estimators/estimator.py)

## TODO:
* More GAN-type implementations
* Create GanSpec (similar to [EstimatorSpec](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/estimators/model_fn.py))
