# Copyright 2020 FiveAI Ltd.
# All rights reserved.
#
# This file is part of the PaRoT toolbox, and is released under the
# "MIT License Agreement". Please see the LICENSE file that should
# have been included as part of this package.

import tensorflow as tf
import numpy as np

from parot.domains import HZ
from .property import Property


class Brightness(Property):
    """
    Encodes a brightness change domain. In terms of HZ, all of
    the entries in the input vector are raised or lowered together
    by epsilon.

    Attributes:
        epsilon (float or tf.Tensor): distance to the input
    """

    SUPPORTED_DOMAINS = [HZ]

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def generate_property(self, domain, input_tensor):
        return domain(
            input_tensor,
            tf.zeros_like(input_tensor),
            tf.ones_like(input_tensor) * self.epsilon)


class UniformChannel(Property):
    """
    Allow each slice of the last axis to vary independently. So for
    example in an image this means that each colour channel can change
    separately but all pixels change together.

    Attributes:
        epsilon (float or tf.Tensor): distance to the input
    """

    SUPPORTED_DOMAINS = [HZ]

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def generate_property(self, domain, input_tensor):
        C = input_tensor.shape[-1]
        E = tf.ones_like(input_tensor[..., :1]) * self.epsilon
        multiples = [1 for i in range(0, len(input_tensor.shape) - 1)] + [C]
        E = tf.tile(E, multiples)
        return domain(input_tensor, tf.zeros_like(input_tensor), E)


class Fourier(Property):
    """
    Creates a domain object consisting of various fourier terms.

    Attributes:
        epsilon (float or tf.Tensor): distance to the input
        BH (int): period of lowest freq wave in vertical direction
        BW (int): period of lowest freq wave in horizontal direction
        FH (int): vertical number of fourier terms, not including a
            constant term
        FW (int): horizontal number of fourier terms
        independent_channels (bool): if True, then it will copy the
            fourier terms for each channel in the input.
    """

    SUPPORTED_DOMAINS = [HZ]

    def __init__(self, epsilon, BH=30, BW=10, FH=5, FW=5,
                 independent_channels=False):
        self.epsilon = epsilon
        self.BH = BH
        self.BW = BW
        self.FH = FH
        self.FW = FW
        self.independent_channels = independent_channels

    def generate_property(self, domain, input_tensor):
        terms = []
        (B, H, W, C) = input_tensor.shape
        for i in range(-self.FH, self.FH):
            for j in range(-self.FW, self.FW):
                kx = 2 * np.pi * i / self.BH
                ky = 2 * np.pi * j / self.BW

                terms.append(
                    [[np.cos(kx * x + ky * y) for y in range(0, W)] for
                     x in range(0, H)])

                if not (i == 0 and j == 0):
                    terms.append(
                        [[np.sin(kx * x + ky * y) for y in range(0, W)]
                         for x in range(0, H)])

        P = len(terms)
        E = tf.constant(terms)
        E = tf.expand_dims(E, axis=-1)
        E = tf.tile(E, [1, 1, 1, C])
        if self.independent_channels:
            E = tf.matrix_diag(E)
            E = tf.transpose(E, [4, 0, 1, 2, 3])
            E = tf.reshape(E, [C * P, 1, H, W, C])
            E = tf.tile(E, [1, B, 1, 1, 1])
            E = tf.reshape(E, [C * P * B, H, W, C])
        else:
            E = tf.expand_dims(E, axis=1)
            E = tf.tile(E, [1, B, 1, 1, 1])
            E = tf.reshape(E, [P * B, H, W, C])

        return domain(
            input_tensor, tf.zeros_like(input_tensor), E * self.epsilon)
