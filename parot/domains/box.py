# Copyright 2020 FiveAI Ltd.
# All rights reserved.
#
# This file is part of the PaRoT toolbox, and is released under the
# "MIT License Agreement". Please see the LICENSE file that should
# have been included as part of this package.

from itertools import count

import tensorflow as tf

from parot.utils import graph
from .domain import Domain


class Box(Domain):
    """
    Abstract domain for an axis aliged box. Specified by giving a center
    position and the halfwidths in each axis.

    Attributes:
        id (int): count of the instance of box for the purpose of variable
            naming
        b (tf.Tensor): halfwidth tensor
        c (tf.Tensor): centre point tensor
    """
    _ids = count(0)

    def __init__(self, c, b):
        self.id = next(self._ids)
        self.c = tf.identity(c, name='box_{}_c'.format(self.id))
        self.b = tf.identity(b, name='box_{}_b'.format(self.id))

    def of_monotone(self, f):
        """
        Given a monotone function `f` computes the abstract transformation
        of `f` on `self`.

        Args:
            f (function): monotone function

        Returns:
            Box: transformation of `f`
        """
        up = f(self.c + self.b)
        lo = f(self.c - self.b)
        return Box((up + lo) / 2, (up - lo) / 2)

    def relu(self):
        """
        relu activation of the Box

        Returns:
            Box: the Box after the relu activation
        """
        return self.of_monotone(tf.nn.relu)

    def __add__(self, x2):
        """
        Sum to a box or a tensor

        Args:
            x2 (Box or tf.Tensor)

        Returns:
            Box: resultant box
        """
        x1 = self
        if isinstance(x2, Box):
            return Box(x1.c + x2.c, x1.b + x2.b)
        elif isinstance(x2, tf.Tensor):
            return Box(x1.c + x2, x1.b)
        else:
            return NotImplemented

    __radd__ = __add__

    def __matmul__(self, x2):
        """
        Matrix multiplication with a tensor

        Args:
            x2 (tf.Tensor)

        Returns:
            Box: resultant box
        """
        x1 = self
        if isinstance(x2, Box):
            return NotImplemented
        elif isinstance(x2, tf.Tensor):
            return Box(x1.c @ x2, x1.b @ abs(x2))
        else:
            return NotImplemented

    def __rmatmul__(self, x1):
        """
        Right-matrix multiplication with a tensor

        Args:
            x1 (tf.Tensor)

        Returns:
            Box: resultant box
        """
        x2 = self
        if isinstance(x1, tf.Tensor):
            return Box(x1 @ x2.c, abs(x1) @ x2.b)
        else:
            return NotImplemented

    def __mul__(self, x2):
        """
        multiplication with a tensor

        Args:
            x2 (tf.tensor)

        Returns:
            Box: resultant box
        """
        x1 = self
        if isinstance(x2, tf.Tensor):
            return Box(x1.c * x2, x1.b * abs(x2))
        else:
            return NotImplemented
    __rmul__ = __mul__

    def __neg__(self):
        """
        Negation of box

        Returns:
            Box: resultant box
        """
        return Box(-self.c, self.b)

    def __sub__(self, x2): return self + (- x2)
    __rsub__ = __sub__

    def __truediv__(self, x2):
        """
        Division with a tensor

        Args:
            x2 (tf.Tensor)

        Returns:
            Box: resultant box
        """
        x1 = self
        if isinstance(x2, tf.Tensor):
            return Box(x1.c / x2, x1.b / abs(x2))
        else:
            return NotImplemented

    def get_center(self): return self.c

    def get_errors(self): return self.b

    def get_parameters(self): return [self.b.shape]

    def get_adversary(self, y_true, n_classes):
        """
        Draws an axis-aligned box around the domain element and then finds the
        point in this box which is furthest from the tensor 'y_true'

        Args:
            y_true (tf.Tensor): the true output
            n_classes (int): number of classes in the output

        Returns:
            tf.Tensor: the furthest tensor from `y_true` in the box
        """
        c = self.get_center()
        s = self.get_errors()
        g = (tf.cast(tf.one_hot(y_true, depth=n_classes), tf.float32) * 2 - 1)
        return c - s * g

    def evaluate(self, p):
        """Evaluate the box at p

        Args:
            p (tf.Tensor): input tensor representing a point to be evaluated

        Returns:
            tf.Tensor: box at p
        """
        (beta,) = p
        beta = tf.clip_by_value(beta, -1.0, 1.0)
        return self.c + self.b * beta

    @staticmethod
    def transform_op(op, inputs):
        """
        Transform operation for a given set of inputs

        Args:
            op (tf.Operation): operation to transform
            inputs (List[tf.Tensor]): list of inputs

        Returns:
            Box: resulting box
        """
        def monotone(op, x):
            return [Box.of_minmax(
                graph.clone_op(op, [x.c + x.b], suffix="_u")[0],
                graph.clone_op(op, [x.c - x.b], suffix="_l")[0]
            )]

        op_type = op.type

        if op_type in ["Shape", "ZerosLike", "OnesLike", "GreaterEqual"]:
            return [None]

        if op_type in ["Transpose", "Reshape", "StridedSlice", "Pack",
                       "ConcatV2"]:
            outputsc = graph.clone_op(op, [i.c if isinstance(i, Box) else i for
                                           i in inputs], suffix="_c")
            outputsb = graph.clone_op(op, [i.b if isinstance(i, Box) else i for
                                           i in inputs], suffix="_b")
            return [Box(c, b) for c, b in zip(outputsc, outputsb)]

        if op_type == "Conv2D":
            [input, filter] = inputs
            [c] = graph.clone_op(op, [input.c, filter], suffix="_c")

            # any other combinations are not supported at this time.
            if not (isinstance(input, Box) and isinstance(filter, tf.Tensor)):
                raise NotImplementedError(
                    'graph includes a non-supported Conv2D ' +
                    'due to its input or filter type (should ' +
                    'be Box and tf.Tensor, respectively)')

            [b] = graph.clone_op(op, [input.b, abs(filter)], suffix="_b")
            return [Box(c, b)]

        # BiasAdd has an extra check that the types are the same.
        if op_type in ["Add", "BiasAdd"]:
            [a, b] = inputs
            return [a + b]

        if op_type == "Mul":
            [a, b] = inputs
            return [a * b]

        if op_type == "MatMul":
            x, y = inputs
            return [x @ y]

        if op_type == "Sub":
            [a, b] = inputs
            return [a - b]

        if op_type == "RealDiv":
            [a, b] = inputs
            return [a / b]

        if op_type == "Neg":
            return [- inputs[0]]

        if op_type in ["MaxPool", "Sigmoid", "Relu"]:
            return monotone(op, inputs[0])

        if op_type == "Softmax":
            return [inputs[0].softmax()]

        raise NotImplementedError(
            ('the {} operation is not implemented by default in HZ for this ' +
             'version of parot').format(op_type))

    def clip_by_value(self, min=0.0, max=1.0):
        """
        clips box between min and max

        Args:
            min (float, optional): minimum bound
            max (float, optional): maximum bound

        Returns:
            Box: clipped box
        """
        return self.of_monotone(lambda x: tf.clip_by_value(x, min, max))

    @staticmethod
    def of_minmax(mn, mx):
        """
        Return a box between mn and mx

        Args:
            mn (tf.Tensor): minimum limit
            mx (tf.Tensor): maximum limit

        Returns:
            Box: resultant box
        """
        return Box((mx + mn) / 2.0, abs(mx - mn) / 2.0)

    def softmax(self):
        """
        Perform a softmax on the last axis.

        ### Implementation:

        Use the observation that:
        ```
        softmax(x)[i]
            = exp(x[i]) / \Sigma j, exp(x[j])
            = exp(x[i]) / (exp(x[i]) + \Sigma (j != i), exp(x[j]))
            = 1 / (1 + \Sigma (j != i), exp(x[j]) / exp(x[i]))
            = \sigma(- log(\Sigma (j != i), exp(x[j]) / exp(x[i])))
            = \sigma(x[i] - log(\Sigma (j != i), exp(x[j])))
            = \sigma(x[i] - LSE (j != i), x[j])
        ```
        Where `LSE := log o \Sigma o exp`.
        And the upper bound for the `i`th component on softmax(x) is found
        by adding the error to `x_i` but subtracting from the others.
        Thus the lower and upper bounds `softmax_l`,`softmax_u` become:
        ```
        softmax_l[i] = \sigma((c - s)[i] - LSE (j != i), (c + s)[j])
        softmax_u[i] = \sigma((c + s)[i] - LSE (j != i), (c - s)[j])
        ```
        Where `s` is the box error and `c` is the center.

        Returns:
            Box: resultant box
        """
        c, s = self.get_center_errors()
        up, lo = c + s, c - s

        shape = tf.shape(c)
        channels = shape[-1]
        rest = shape[:-1]
        rank = len(list(c.shape))
        multiples = [1 for i in range(0, rank - 1)] + [channels]
        llshape = tf.concat([rest, [channels, channels - 1]], axis=0)

        mask = tf.fill([channels, channels], True)
        mask = tf.linalg.set_diag(mask, tf.fill([channels], False))
        mask = tf.reshape(mask, [-1])

        ll = tf.reduce_logsumexp(tf.reshape(tf.boolean_mask(
            tf.tile(lo, multiples), mask, axis=(rank - 1)), llshape), axis=-1)
        uu = tf.reduce_logsumexp(tf.reshape(tf.boolean_mask(
            tf.tile(up, multiples), mask, axis=(rank - 1)), llshape), axis=-1)

        smu = tf.sigmoid(up - ll)
        sml = tf.sigmoid(lo - uu)
        return Box.of_minmax(smu, sml)
