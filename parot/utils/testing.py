# Copyright 2020 FiveAI Ltd.
# All rights reserved.
#
# This file is part of the PaRoT toolbox, and is released under the
# "MIT License Agreement". Please see the LICENSE file that should
# have been included as part of this package.

import tensorflow as tf

from parot.utils import graph


def PGD(property, domain, epsilon=0.01, step_count=200, learning_rate=0.5,
        domain_max=1.0, domain_min=0.0,
        ):
    """
    Computes an adversarial example using PGD traversing the graph between the
    input, model and loss.

    Args:
        property (parot.properties.Property): the property to test
        domain (parot.domains.Domain): abstraction domain to use
        epsilon (float, optional): the diameter of the attack
        step_count (int, optional): number of steps
        learning_rate (float, optional): learning rate
        domain_max (tf.Tensor, optional): the max value that the domain is
         allowed to inhabit: bound to clip the result of `self.property`
        domain_min (tf.Tensor, optional): the min value that the domain is
         allowed to inhabit: bound to clip the result of `self.property`

    Returns:
        Callable: returns f
    """
    opt = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=learning_rate)

    def f(x, pred, loss):
        """
        PGD function f

        Args:
            x (tf.Tensor): input tensor
            pred (List[tf.Tensor]): list of tensors to be added to the graph
            loss (tf.Tensor): tensor version of the loss

        Returns:
            Tuple[tf.Tensor]: (adversary, adversary_pred, adversary_loss)
            where adversary are the input tensor permuted to maximise the loss
            adversary_pred are the prediction values for the input adversary
            adversary_loss is a scalar saying what the loss is.
        """
        x_domain = property(epsilon).of(domain, x)
        attack_shape = x_domain.get_parameters()

        def clip_fn(Delta_s):
            x = x_domain.evaluate([tf.convert_to_tensor(Delta) for Delta in
                                   Delta_s])

            # can't use tf.clip_by_value since min and max might be tensors
            x = tf.math.maximum(x, domain_min)
            x = tf.math.minimum(x, domain_max)
            return x

        Delta_s = [tf.Variable(initial_value=tf.zeros(sh), trainable=False) for
                   (i, sh) in enumerate(attack_shape)]
        # can also initialise at zero.
        init_ops = [tf.compat.v1.assign(Delta, tf.random.uniform(
            tf.shape(Delta)) * 2 - 1) for Delta in Delta_s]

        def body(i):
            adversary = clip_fn(Delta_s)
            [cloned_loss] = graph.clone_subgraph(outputs=[loss], inputs=[x],
                                                 new_inputs=[adversary])
            train_op = opt.minimize(-cloned_loss, var_list=Delta_s)
            with tf.control_dependencies([train_op]):
                assign_ops = [
                    tf.compat.v1.assign(Delta, tf.clip_by_value(Delta, -1, 1))
                    for Delta in Delta_s]
            with tf.control_dependencies(assign_ops):
                return i + 1

        counter = tf.Variable(0, name="counter")

        with tf.control_dependencies(init_ops):
            z = tf.while_loop(lambda i: i < step_count, body, [counter],
                              name='PGDInitilization')

        with tf.control_dependencies([z]):
            adversary = clip_fn([Delta.read_value() for Delta in Delta_s])
            [adversary_loss, *adversary_pred] = graph.clone_subgraph(
                outputs=[loss] + [r for r in pred],
                inputs=[x], new_inputs=[adversary]
            )

        return (adversary, adversary_pred, adversary_loss)

    return f
