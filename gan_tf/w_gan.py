"""Provides Wasserstein GAN class WGan."""
import tensorflow as tf
from gan_tf.gan import Gan


class WGan(Gan):
    """
    Wasserstein GAN base class.

    This base implementation is valid but unlikely to converge. See other
    implementations for better implementations.

    Implementations:
        WeightClippedWGan
        GradientPenalizedWGan
    """

    def get_losses(
            self, real_logits, fake_logits, real_samples, fake_samples, mode):
        """
        Get losses associated with this GAN.

        Inputs:
            real_logits: logits associated with real inputs
            fake_logits: logits associated with fake inputs
        Returns:
            (c_loss, g_loss)
            c_loss: critic loss
            g_loss: generator loss
        """
        c_fake_loss = tf.reduce_mean(fake_logits)
        g_loss = -c_fake_loss
        c_real_loss = -tf.reduce_mean(real_logits)
        c_loss = c_fake_loss + c_real_loss

        tf.summary.scalar('c_loss_real', c_real_loss)
        tf.summary.scalar('c_loss_fake', c_fake_loss)
        return c_loss, g_loss


class WeightClippedWGan(WGan):
    """WGan implementation with weight clipping."""

    def __init__(self, *args, **kwargs):
        """Redirecting constructor with different default name."""
        if 'name' not in kwargs:
            kwargs['name'] = 'wgan-wc'
        super(WeightClippedWGan, self).__init__(*args, **kwargs)

    def get_train_ops(self, real_logits, fake_logits, global_step):
        """
        Get operations for training critic and generator.

        Returns:
            c_ops: critic train ops. Must contain at least loss and opt
            g_opt: generator train ops. Must contain at least loss and opt.

        """
        c_ops, g_ops = super(WGan, self).get_train_ops(
            real_logits, fake_logits, global_step)

        critic_vars = self.critic_vars()
        clip_val = self._params['max_critic_var'] \
            if 'max_critic_var' in self._params else 1e-2
        c_clip = [p.assign(tf.clip_by_value(p, -clip_val, clip_val))
                  for p in critic_vars]
        c_ops['clip'] = c_clip
        return c_ops, g_ops


class GradientPenalizedWGan(WGan):
    """
    WGan implementation with L2 loss factor on critic gradients.

    See Gulrajani et al for details, https://arxiv.org/pdf/1704.00028.pdf .
    """

    def __init__(self, *args, **kwargs):
        """Redirecting constructor with different default name."""
        if 'name' not in kwargs:
            kwargs['name'] = 'wgan-gp'
        super(GradientPenalizedWGan, self).__init__(*args, **kwargs)

    def get_losses(
            self, real_logits, fake_logits, real_samples, fake_samples, mode):
        """
        Get losses associated with this GAN.

        Inputs:
            real_logits: logits associated with real inputs
            fake_logits: logits associated with fake inputs
        Returns:
            (c_loss, g_loss)
            c_loss: critic loss
            g_loss: generator loss
        """
        c_loss, g_loss = super(GradientPenalizedWGan, self).get_losses(
            real_logits, fake_logits, real_samples, fake_samples, mode)
        batch_size = self._params['batch_size']
        eps_shape = [batch_size] + [1]*(len(real_samples.shape)-1)
        eps = tf.random_uniform(shape=eps_shape, name='eps')
        mixed_samples = eps * real_samples + (1 - eps) * fake_samples
        # mixed_logits = self.get_scoped_critic_logits(
        #     mixed_samples, mode=mode, reuse=True)
        mixed_logits = self._call_critic_logits_fn(
            mixed_samples, mode=mode, reuse=True)

        grad = tf.gradients(mixed_logits, mixed_samples)[0]
        norm_axes = range(1, len(grad.shape))
        grad_n = tf.sqrt(tf.reduce_sum(grad**2, axis=norm_axes))
        grad_loss = tf.reduce_sum((grad_n - 1.0)**2)
        tf.summary.scalar('c_loss_logits', c_loss)
        c_loss = c_loss + self._params['gradient_loss_factor']*grad_loss
        tf.summary.scalar('c_loss_grad', grad_loss)
        return c_loss, g_loss
        # c_fake_loss = tf.reduce_mean(fake_logits)
        # g_loss = -c_fake_loss
        # c_real_loss = -tf.reduce_mean(real_logits)
        #
        # fake_grads = tf.gradients(c_fake_loss, fake_samples)
        # real_grads = tf.gradients(c_real_loss, real_samples)
        # grads = tf.stack([fake_grads, real_grads], axis=0)
        # c_grad_loss = (tf.reduce_sum(grads**2) - 1)**2
        #
        # params = self.params
        # grad_loss_factor = params['grad_loss_factor'] if \
        #     'grad_loss_factor' in params else 10.
        #
        # c_loss = tf.add_n(
        #     [c_fake_loss, c_real_loss, grad_loss_factor*(c_grad_loss)])
        #
        # tf.summary.scalar('c_loss_real', c_real_loss)
        # tf.summary.scalar('c_loss_fake', c_fake_loss)
        # tf.summary.scalar('c_loss_grad', c_grad_loss)
        # tf.summary.scalar('c_loss', c_loss)
        # tf.summary.scalar('g_loss', g_loss)
        #
        # return c_loss, g_loss

    @staticmethod
    def _default_params():
        params = WGan._default_params()
        params.update({
            'beta1': 0.0,
            'beta2': 0.9,
            'gradient_loss_factor': 10.0,
        })
        return params

    def get_critic_opt(self, critic_loss, critic_vars, global_step):
        """Get critic optimization operation."""
        c_lr = self._params['critic_learning_rate']
        beta1 = self._params['beta1']
        beta2 = self._params['beta2']
        critic_opt = tf.train.AdamOptimizer(
            c_lr, beta1=beta1, beta2=beta2).minimize(
                critic_loss, var_list=critic_vars, global_step=global_step)
        return critic_opt

    def get_generator_opt(self, generator_loss, generator_vars, global_step):
        """Get critic optimization operation."""
        g_lr = self._params['generator_learning_rate']
        beta1 = self._params['beta1']
        beta2 = self._params['beta2']
        generator_opt = tf.train.AdamOptimizer(
            g_lr, beta1=beta1, beta2=beta2).minimize(
                generator_loss, var_list=generator_vars,
                global_step=global_step)
        return generator_opt
