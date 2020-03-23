# Copyright 2020 FiveAI Ltd.
# All rights reserved.
#
# This file is part of the PaRoT toolbox, and is released under the
# "MIT License Agreement". Please see the LICENSE file that should
# have been included as part of this package.

import tensorflow as tf

from parot.domains import HZ, Box
from .property import Property


class Ball(Property):
    """
    l_\infty epsilon-ball property around the input

    Attributes:
        epsilon (float or tf.Tensor): distance to the input
    """

    SUPPORTED_DOMAINS = [Box, HZ]

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def generate_property(self, domain, input_tensor):
        box_obj = Box(input_tensor, self.epsilon * tf.ones_like(input_tensor))

        if domain == Box:
            return box_obj
        elif domain == HZ:
            return BallPromoted(self.epsilon).of(domain, input_tensor)


class BallDemoted(Property):
    """
    Epsilon ball in the zonotope such that no component is promoted. That is,
    in an HZ, the `E` matrix is zero.

    Args:
        epsilon (float or tf.Tensor): scalar or tensor
    """

    SUPPORTED_DOMAINS = [HZ]

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def generate_property(self, domain, input_tensor):
        Cs = input_tensor.shape[1:]
        return domain(
            input_tensor,
            tf.ones_like(input_tensor) * self.epsilon,
            tf.zeros([0, *Cs]))


class BallPromoted(Property):
    """
    Epsilon ball in the zonotope such that all components are promoted. That
    is, in an HZ, the axis-aligned `b` vector is zero and each component of
    `epsilon` is converted to a column of the `E` matrix.

    Args:
        epsilon (float or tf.Tensor): scalar or tensor
    """

    SUPPORTED_DOMAINS = [HZ]

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def generate_property(self, domain, input_tensor):
        return BallDemoted(self.epsilon).of(domain, input_tensor).promote_all()
