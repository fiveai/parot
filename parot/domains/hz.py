# Copyright 2020 FiveAI Ltd.
# All rights reserved.
#
# This file is part of the PaRoT toolbox, and is released under the
# "MIT License Agreement". Please see the LICENSE file that should
# have been included as part of this package.

from functools import reduce
from itertools import count
import warnings

import tensorflow as tf

from .domain import Domain
from parot.utils import graph, util

relu = tf.nn.relu


class HZ(Domain):
    """
    A hybrid zonotope on a shape `p = (p1,p2,...)` is a set characterised
    by the tensors `c`, `b` and `E`. The set `Z` is characterised as
    `Z := {c + b^T \beta  + E e | \beta \in [-1,1]^p, e \in [-1,1]^m}`.

    ### Implementation notes:

    A batch dimension should always be provided, even if only set to one.
    The first dimension in the shape is assumed to be the batch dimension.

    To account for batch sizes greater than one and to keep some methods
    efficient, the `e` dimension in `E` is folded in to the batch
    dimension. This means that `c`,`b`,`E` all have the same rank. To get
    a version of `E` with the batch and `e` dimensions unflattened call
    `get_E_unflattened`.

    Attributes:
        id (int): count of the instance of HZ for the purpose of variable
            naming
        b (tf.Tensor): the on-axis uncorrelated errors
        c (tf.Tensor): center of the object
        E (tf.Tensor): zonotope errors
    """
    _ids = count(0)

    def __init__(self, c, b, E):
        # we store the error dimension flattened along with batch-size to make
        # the non-activation transormations simpler.
        self.id = next(self._ids)
        self.c = tf.identity(c, name='hz_{}_c'.format(self.id))
        self.b = tf.identity(b, name='hz_{}_b'.format(self.id))
        self.E = tf.identity(E, name='hz_{}_E'.format(self.id))
        if self.E.shape[0] is None:
            raise ValueError('E should have a defined shape')

        if len(self.E.shape) != len(self.c.shape):
            raise ValueError('E and c have different shapes')

    def get_center(self): return self.c

    def get_error_dim(self): return self.E.shape[0] // self.c.shape[0]

    def get_batch_dim(self): return self.c.shape[0]

    def get_space_dims(self):
        """
        Return the shape of `self.c` without the batch dimension.

        Returns:
            tf.Tensor: space dimensions
        """
        return self.c.shape[1:]

    def has_E_matrix(self):
        """Returns True if the HZ has an E matrix

        Returns:
            bool: True if the instance has an E matrix
        """
        return self.E.shape[0] != 0

    def get_E_unflattened(self):
        """
        E is stored with the error dimension reshaped in to the batch
        dimension to make some calculations simpler.
        `self.E : (error_dim * batch_dim, ..rest)`.
        `self.get_E_unflattened() : (error_dim, batch_dim, ..rest)`

        Returns:
            tf.Tensor: unflattened E matrix
        """
        # this operation will fail if any dimensions is unknown
        if any([dim._value is None for dim in self.c.shape]):
            raise ValueError(
                'The implementation of HZ requires all dimensions of ' +
                'all tensors to be known. If the undefined dimension is ' +
                'due to a fixed batch size, consider reshaping your input ' +
                'to include that batch size')

        return tf.reshape(self.E, [-1, *self.c.shape])

    def get_errors(self):
        """
        Return the zonotope error vector

        Returns:
            tf.Tensor: vector of errors
        """
        E = self.get_E_unflattened()
        return self.b + tf.reduce_sum(abs(E), axis=0)

    def get_parameters(self):
        """
        Return the zonotope parameters

        Returns:
            tf.Tensor: parameters
        """
        p = self.get_error_dim()
        b = self.get_batch_dim()
        rest = self.get_space_dims()
        return ([b] + list(rest), [p, b])

    def get_adversary(self, y_true, n_classes):
        """
        Draws an arbitrary box around the domain element and then finds the
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
        """Evaluate the zonotope at p

        Args:
            p (tf.Tensor): input tensor representing a point to be evaluated

        Returns:
            tf.Tensor: zonotope at p
        """
        (beta, e) = p
        E = self.get_E_unflattened()
        [p, b, *rest] = E.shape
        beta = tf.clip_by_value(beta, -1, 1)
        e = tf.clip_by_value(e, -1, 1)
        e = tf.reshape(e, [p, b] + [1 for r in rest])
        E = E * e
        E = tf.reduce_sum(E, axis=0)
        return self.c + self.b * beta + E

    def is_inside(self, x):
        """
        Check if the point `x` is inside the zonotope.
        Showing that x lies within a zonotope is a linear programming problem.
        For our purposes we can just overapproximate to a box and check if
        it lies in the box.

        Args:
            x (tf.Tensor): tensor to check

        Returns:
            bool: whether x is inside the zonotope
        """
        return tf.reduce_all(self.get_errors() >= abs(x - self.c))

    def box_relu(self):
        """
        Box version of relu transformer for comparison with `self.relu()`.

        Returns:
            HZ: a transformed zonotope
        """
        c = self.c
        s = self.get_errors()
        return HZ.of_minmax(relu(c + s), relu(c - s))

    def box_sigmoid(self): return self.box_monotone(tf.math.sigmoid)

    def sigmoid(self):
        """
        Computes the transformation of a zonotope through a sigmoid
        activation function. It works by drawing the smallest
        parallelogram about the sigmoid graph.

        Returns:
            HZ: a transformed zonotope
        """
        def sigma(x): return tf.math.sigmoid(x)
        c = self.c
        s = self.get_errors()
        up = c + s
        lo = c - s
        meu = tf.div_no_nan(sigma(up) - sigma(lo), s) / 2

        # to find extrema we wish to solve  `\delta_x \sigma(x) = \meu`
        # Let `Y = exp(-x)`. Then `\delta_x Y = - Y`, `\sigma(x) = 1 / (1 + Y)`
        # and `\delta_x \sigma(x) = Y / (1 + Y)^2`
        # So we need `\meu + (2\meu - 1)Y + \meu Y^2 = 0`.
        # Which happens when `Y = b +- sqrt(b^2 - 1)`
        # Where `b := (\meu^{-1} - 2) / 2`.

        b = tf.div_no_nan(1.0, meu + 0.01) / 2 - 1
        # INF \geq b \geq 1 avoids nans in gradient

        b = tf.maximum(b, 1.00)
        delta = tf.sqrt(b * b - 1)

        # x1, x2 are the input points at which the error is maximised.
        x1 = - tf.log(b + delta)
        x1 = tf.where(tf.is_finite(x1), x1, tf.zeros_like(x1))
        x2 = - tf.log(b - delta)
        x2 = tf.where(tf.is_finite(x2), x2, tf.zeros_like(x2))
        x1 = tf.maximum(tf.minimum(x1, up), lo)
        x2 = tf.maximum(tf.minimum(x2, up), lo)
        eps1 = sigma(x1) - sigma(lo) - meu * (x1 - lo)
        eps2 = sigma(x2) - sigma(lo) - meu * (x2 - lo)
        eps_mx = relu(tf.maximum(eps1, eps2))
        eps_mn = relu(tf.maximum(-eps1, -eps2))
        epsilon = (eps_mx + eps_mn) / 2
        c = ((sigma(up) + eps_mx) + (sigma(lo) - eps_mn)) / 2
        b = self.b * meu + epsilon
        E = self.get_E_unflattened() * meu
        E = tf.reshape(E, tf.shape(self.E))
        return HZ(c, b, E)

    def __add__(self, x2):
        """
        Add a zonotope or a tensor or a float

        Args:
            x2 (HZ or tf.Tensor or float)

        Returns:
            HZ: resultant zonotope
        """
        x1 = self
        if isinstance(x2, tf.Tensor) or isinstance(x2, float):
            return HZ(x1.c + x2, x1.b, x1.E)
        elif isinstance(x2, HZ) and (x1.get_error_dim() == x2.get_error_dim()):

            print("[WARNING]: adding two hybrid zonotopes", x1.c.name, ",",
                  x2.c.name,
                  ". Assuming that the E-matrix error terms are correlated.")
            return HZ(x1.c + x2.c, x1.b + x2.b, x1.E + x2.E)
        else:
            return NotImplemented
    __radd__ = __add__

    def __sub__(self, x2):
        """
        Subtracts a zonotope or a tensor or a float

        Args:
            x2 (HZ or tf.Tensor or float)

        Returns:
            HZ: resultant zonotope
        """
        x1 = self
        if isinstance(x2, tf.Tensor) or isinstance(x2, float):
            return HZ(x1.c - x2, x1.b, x1.E)
        elif isinstance(x2, HZ):
            return x1 + (- x2)
        else:
            return NotImplemented

    def __neg__(self): return HZ(- self.c, self.b, self.E)

    def __mul__(self, x2):
        """
        Multiplication with a zonotope or a tensor or a float

        Args:
            x2 (HZ or tf.Tensor)

        Returns:
            HZ: resultant zonotope
        """
        x1 = self
        if isinstance(x2, tf.Tensor) or isinstance(x2, float):
            return HZ(x1.c * x2, x1.b * abs(x2), x1.E * x2)
        if isinstance(x2, HZ):
            return x1.mul_same_error(x2)
        return NotImplemented
    __rmul__ = __mul__

    def __truediv__(self, x2):
        """
        Division with a zonotope or a tensor

        Args:
            x2 (HZ or tf.Tensor)

        Returns:
            HZ: resultant zonotope
        """
        x1 = self
        if isinstance(x2, tf.Tensor):
            return HZ(x1.c / x2, x1.b / abs(x2), x1.E / x2)
        elif isinstance(x2, HZ):
            return x1.div_same_error(x2)
        else:
            return NotImplemented

    def minimum(self, x2):
        """
        Minimum of the HZ instance and x2

        Args:
            x2 (HZ or tf.Tensor or float)

        Returns:
            HZ: resultant zonotope
        """
        x1 = self
        if isinstance(x2, tf.Tensor) or isinstance(x2, float):
            return - (- x1 + x2).relu() + x2
        elif isinstance(x2, HZ):
            return - (-x1).max_same_error(-x2)
        else:
            return NotImplemented

    def maximum(self, x2):
        """
        Maximum of the HZ instance and x2

        Args:
            x2 (HZ or tf.Tensor or float)

        Returns:
            HZ: resultant zonotope
        """
        x1 = self
        if isinstance(x2, tf.Tensor) or isinstance(x2, float):
            return (x1 - x2).relu() + x2
        elif isinstance(x2, HZ):
            return x1.max_same_error(x2)
        else:
            return NotImplemented

    def mul_same_error(self, y):
        """
        Multiply the HZ's (the HZ instance and y), while maintaing the same
        error.

        Args:
            y (HZ): input HZ to multiply by

        Returns:
            HZ: resultant zonotope
        """
        x = self

        xc, xs = x.get_center_errors()
        yc, ys = y.get_center_errors()

        xu, xl = (xc + xs), (xc - xs)
        yu, yl = (yc + ys), (yc - ys)
        bds = tf.stack([xu * yu, xl * yu, xu * yl, xl * yl], axis=-1)
        up = tf.reduce_max(bds, axis=-1)
        lo = tf.reduce_min(bds, axis=-1)

        return HZ((up + lo) / 2, abs(up - lo) / 2, tf.zeros_like(x.E))

    def div_same_error(self, y):
        """
        Divide the HZ's (the HZ instance and y), while maintaing the same
        error.

        Args:
            y (HZ): input HZ to divide by

        Returns:
            HZ: resultant zonotope
        """
        x = self

        xc, xs = x.get_center_errors()
        yc, ys = y.get_center_errors()

        up = (xc + xs) / relu(yc - ys)
        lo = (xc - xs) / (yc + ys)
        c = (up + lo) / 2

        return HZ(c=c, b=abs(up - lo) / 2, E=tf.zeros_like(x.E))

    def reciprocal(self):
        """
        Return the reciprocal of the HZ instance

        Returns:
            HZ: resultant zonotope
        """
        c = self.get_center()
        s = self.get_errors()
        up = c + s
        lo = c - s

        meu = tf.div_no_nan(
            tf.math.reciprocal(up) - tf.math.reciprocal(lo),
            2.0 * s)
        x = tf.rsqrt(-meu) * tf.sign(c)
        epsilon = util.linterp(
            lo, up,
            tf.math.reciprocal(lo),
            tf.math.reciprocal(up), x) - tf.math.reciprocal(x)

        c = (tf.math.reciprocal(up) + tf.math.reciprocal(lo) - epsilon) / 2
        b = self.b * abs(meu) + abs(epsilon)
        E = self.get_E_unflattened() * meu
        E = tf.reshape(E, tf.shape(self.E))
        return HZ(c, b, E)

    def transform_convex(self, f, extremum_fn):
        """
        Idea; the `extremum_fn` takes the gradient `\meu` of a line
        between two points `l,u` on `f` and gives back an `x` such that
        `df/dx - \meu = 0`.
        This is then used to compute a bounding zonotope for the function.
        All of this works because we know that `f` is either convex or concave.

        Args:
            f (Callable): function to transform
            extremum_fn (function): extremum function

        Returns:
            HZ: resultant zonotope
        """
        c, s = self.get_center_errors()
        up = c + s
        lo = c - s

        meu = tf.div_no_nan(f(up) - f(lo), 2.0 * s)
        x = extremum_fn(meu)
        x = tf.clip_by_value(x, lo, up)
        epsilon = util.linterp(lo, up, f(lo), f(up), x) - f(x)

        c = (f(up) + f(lo) - epsilon) / 2
        b = self.b * meu + (abs(epsilon) / 2.0)
        E = self.get_E_unflattened() * meu
        E = tf.reshape(E, tf.shape(self.E))
        return HZ(c, b, E)

    def relu(self):
        """
        Pass the zonotope through relu activation.

        Returns:
            HZ: resultant zonotope
        """
        if self.has_E_matrix():
            return self.box_relu()
        else:
            return self.transform_convex(relu, tf.zeros_like)

    def abs(self): return self.transform_convex(tf.abs, tf.zeros_like)

    def exp(self): return self.transform_convex(tf.exp, tf.log)

    def log(self): return self.transform_convex(tf.log, tf.reciprocal)

    def log1p(self):
        """
        Element-wise natural logarithm of (1+x)

        Returns:
            HZ: resultant zonotope
        """
        return self.transform_convex(tf.log1p, lambda meu: tf.reciprocal(
            meu) - 1)

    def reduce(self, op, axis=-1):
        """
        Apply the original operation while reducing an axis.

        Args:
            op (tf.Operation): operation to reduce
            axis (int, optional): axis specifier

        Returns:
            HZ: resultant zonotope
        """
        if (axis == 0):
            raise NotImplementedError(
                "axis=0 not implemented for reduction operation.")
        return HZ(
            c=op(self.c, axis=axis),
            b=op(self.b, axis=axis),
            E=op(self.E, axis=axis)
        )

    def reduce_sum(self, axis=-1): return self.reduce(tf.reduce_sum, axis)

    def reduce_mean(self, axis=-1): return self.reduce(tf.reduce_mean, axis)

    def expand_dims(self, axis=-1): return self.reduce(tf.expand_dims, axis)

    def box_max(self, y):
        """
        A simple Box maxing procedure. Using `max_same_error` should give
        smaller errors.

        Args:
            y (HZ): other HZ to max in terms of box

        Returns:
            HZ: resultant zonotope
        """
        x = self
        xs = x.get_errors()
        ys = y.get_errors()
        up = tf.maximum(x.c + xs, y.c + ys)
        lo = tf.maximum(x.c - xs, y.c - ys)
        return HZ.of_minmax(up, lo)

    def op_box_monotone(self, op):
        """
        Assuming the given `op` is monotone, will compute the axis-aligned
        bounding box transformer for `op`.

        Args:
            op (tf.Operation): tensor describing an operation

        Returns:
            HZ: resultant zonotope
        """
        s = self.get_errors()
        c = self.c
        return HZ.of_minmax(
            graph.clone_op(op, [c + s], suffix="_u")[0],
            graph.clone_op(op, [c - s], suffix="_l")[0]
        )

    def box_monotone(self, f):
        """
        Apply a monotone function f to the HZ.

        Args:
            f (Callable): Description

        Returns:
            HZ: Description
        """
        c, s = self.get_center_errors()
        return HZ.of_minmax(f(c + s), f(c - s))

    def box_softmax(self):
        """
        Apply softmax to the zonotope.

        Returns:
            HZ: resultant zonotope
        """
        c, b = self.to_box().softmax().get_center_errors()
        return HZ(c, b, tf.zeros_like(self.E))

    def max_same_error(self, y):
        """
        Finds the max of two zonotopes assuming that the E matrices are
        using the same error terms.

        Args:
            y (HZ): other HZ to max

        Returns:
            HZ: resultant zonotope
        """
        x = self

        if x.get_error_dim() != y.get_error_dim():
            raise ValueError(
                'the dimensions of the error term in the two HZs should ' +
                'be equal')

        Ex = x.get_E_unflattened()  # : (e,b,h,w,c)
        Ey = y.get_E_unflattened()
        sx = x.b + tf.reduce_sum(abs(Ex), axis=0)  # (b,h,w,c)
        sy = y.b + tf.reduce_sum(abs(Ey), axis=0)
        ux = x.c + sx
        lx = x.c - sx
        uy = y.c + sy
        ly = y.c - sy
        up = tf.maximum(ux, uy)
        lo = tf.maximum(lx, ly)

        meux = tf.div_no_nan(up - tf.maximum(lx, uy), sx) / 2
        meuy = tf.div_no_nan(up - tf.maximum(ly, ux), sy) / 2
        epsilon = (up - 2 * meux * sx - 2 * meuy * sy - lo) / 2

        c = (up + lo) / 2
        b = meux * x.b + meuy * y.b + epsilon
        E = meux * Ex + meuy * Ey  # : (e, b,h,w,c)
        E = tf.reshape(E, tf.shape(x.E))
        return HZ(c, b, E)

    def __matmul__(self, w):
        """
        Matrix multiplication

        Args:
            w (tf.Tensor): matrix to multiply by in tensor form

        Returns:
            HZ: resultant zonotope
        """
        return HZ(
            c=self.c @ w,
            b=self.b @ abs(w),
            E=self.E @ w
        )

    def __rmatmul__(self, l):
        """
        Right-matrix multiplication

        Args:
            l (tf.Tensor): matrix to multiply by in tensor form

        Returns:
            HZ: resultant zonotope
        """
        return HZ(
            c=l @ self.c,
            b=abs(l) @ self.b,
            E=l @ self.E
        )

    def __getitem__(self, key):
        """
        Get a new HZ instance by slicing the individual components of
        the given instance.

        Args:
            key (Slice): key to get in the individual components of HZ

        Returns:
            HZ: resultant zonotope
        """
        if (key[0] != slice(None)):
            raise NotImplementedError(
                "Can't do anything except [:] on the batch dimension.")
        return HZ(self.c.__getitem__(key), self.b.__getitem__(key),
                  self.E.__getitem__(key))

    def max_pool_2x2(self):
        """
        Perform a non-overlapping 2x2 maxpool. Assumes that both height and
        width are an even number.

        Returns:
            HZ: max pooled HZ
        """
        x = self
        x = x[:, :, ::2, :].max_same_error(x[:, :, 1::2, :])
        x = x[:, ::2, :, :].max_same_error(x[:, 1::2, :, :])
        return x

    def concat(self, other, axis):
        """
        Concatenate `other` with the given instance of HZ.

        Args:
            other (HZ or tf.Tensor): either an instance of a HZ or a tensor
                which can be used to generate an instance
            axis (int): axis in which the concatenation occurs

        Returns:
            HZ: resultant zonotope
        """
        if isinstance(other, HZ):
            c = tf.concat([self.c, other.c], axis=axis)
            b = tf.concat([self.b, other.b], axis=axis)
            E1 = self.get_E_unflattened()
            E2 = other.get_E_unflattened()
            [e1, batch_size, *_] = E1.shape
            [e2, _, *_] = E2.shape
            EE1 = tf.concat([E1, tf.zeros_like(E2)], axis=0)
            EE2 = tf.concat([tf.zeros_like(E1), E2], axis=0)
            E = tf.concat([EE1, EE2], axis=axis + 1)
            [_, *cshape] = c.shape
            new_shape = [(e1 + e2) * batch_size, *cshape]
            E = tf.reshape(E, new_shape)
            return HZ(c, b, E)
        else:
            return HZ(
                c=tf.concat([self.c, other], axis=axis),
                b=tf.concat([self.b, tf.zeros_like(other)], axis=axis),
                E=tf.concat([self.E, tf.zeros_like(other)], axis=axis),
            )

    @staticmethod
    def transform_op(op, inputs):
        """
        Transform operation for a given set of inputs

        Args:
            op (tf.Operation): operation to transform
            inputs (List[tf.Tensor]): list of inputs

        Returns:
            HZ: resulting zonotope
        """
        def input_conv(n):
            return [getattr(i, n) if isinstance(i, HZ) else i for i in inputs]

        op_type = op.type

        if op_type in ["Shape", "ZerosLike", "OnesLike", "GreaterEqual"]:
            return [None]

        if op_type == "ConcatV2":
            [z1, z2, axis] = inputs
            if not isinstance(z1, HZ) or isinstance(axis, HZ):
                raise NotImplementedError(
                    'graph includes a non-supported ConcatV2 ' +
                    'due to its first or second input type (should ' +
                    'be HZ and tf.Tensor, respectively)')

            return [z1.concat(z2, axis)]

        if op_type == "Reshape":
            input, _ = inputs

            if not isinstance(input, HZ):
                raise NotImplementedError(
                    'the input to the Reshape operation is not a HZ, which ' +
                    'is not supported in this version of parot')

            output = op.outputs[0]
            Cs = output.shape[1:]
            return [HZ(
                c=tf.reshape(input.c, [-1, *Cs]),
                b=tf.reshape(input.b, [-1, *Cs]),
                E=tf.reshape(input.E, [-1, *Cs]),
            )]

        if op_type in ["Transpose", "StridedSlice"]:
            # perform the operation component-wise
            ocs = graph.clone_op(op, input_conv("c"), suffix="_c")
            obs = graph.clone_op(op, input_conv("b"), suffix="_b")
            oEs = graph.clone_op(op, input_conv("E"), suffix="_E")
            return [HZ(c, b, E) for c, b, E in zip(ocs, obs, oEs)]

        if op_type == "Select":
            cond, y, z = inputs

            if isinstance(cond, HZ):
                raise NotImplementedError(
                    'the first input to the Select operation is a HZ, ' +
                    'which is not supported in this version of parot')

            return [HZ(
                c=graph.clone_op(op, input_conv("c"), suffix="_c")[0],
                b=graph.clone_op(op, input_conv("b"), suffix="_b")[0],
                E=graph.clone_op(op, input_conv("E"), suffix="_E")[0]
            )]

        if op_type in ["Sum", "Mean"]:
            [z, reduction_indices] = inputs
            if len(list(reduction_indices.shape)) == 0:
                # dealing with reductions where `axis` might be (),
                # which means that it should reduce to a scalar.
                e = z.get_error_dim()
                E = tf.reshape(z.E, [e, -1])
                if op_type == "Sum":
                    E = tf.reduce_sum(E, axis=1)
                elif op_type == "Mean":
                    E = tf.reduce_mean(E, axis=1)
            else:
                [E] = graph.clone_op(op, [z.E, reduction_indices], suffix="_E")

            [c] = graph.clone_op(op, input_conv("c"), suffix="_c")
            [b] = graph.clone_op(op, input_conv("b"), suffix="_b")
            return [HZ(c, b, E)]

        if op_type == "MatMul":
            x, w = inputs
            if isinstance(w, HZ):
                raise NotImplementedError(
                    'the second input to the MatMul operation is a HZ, ' +
                    'which is not supported in this version of parot')

            b = x.b
            HW = b.shape[1:]
            b = tf.einsum("...ij,jk->...ijk", b, w)
            perm = [
                *(i + 1 for i in range(len(HW))),
                0,
                *(len(HW) + 1 + i for i in range(len(w.shape) - 1))]
            b = tf.transpose(b, perm=perm)
            b = tf.reshape(b, [-1, *w.shape[1:]])
            c = x.c @ w
            E = x.E @ w
            E = tf.concat([E, b], axis=0)
            return [HZ(c, tf.zeros_like(c), E)]

        if op_type == "Conv2D":
            x, w = inputs

            if isinstance(w, HZ):
                return NotImplementedError(
                    'the filter must be a tf.Tensor in this version of parot')

            [c] = graph.clone_op(op, [x.c, w], suffix="_c")
            [b] = graph.clone_op(op, [x.b, abs(w)], suffix="_b")
            [E] = graph.clone_op(op, [x.E, w], suffix="_E")
            return [HZ(c, b, E)]

        if op_type in ["MaxPool"]:
            warnings.warn("MaxPool is only implemented " +
                          "for `keras.MaxPool2D(2)` at the moment")
            [x] = inputs
            y = x.max_pool_2x2()
            return [y]

        if op_type in ["Add", "BiasAdd"]:
            [a, b] = inputs
            return [a + b]

        if op_type in ["Mul"]:
            [a, b] = inputs
            return [a * b]

        if op_type in ["Sub"]:
            [a, b] = inputs
            if not isinstance(a, HZ):
                return [(-b) + a]
            else:
                return [a - b]

        if op_type in ["RealDiv"]:
            [a, b] = inputs
            return [a / b]

        if op_type in ["Maximum"]:
            [a, b] = inputs
            c = a.maximum(b)
            return [b.maximum(a) if c is NotImplemented else c]

        if op_type in ["Minimum"]:
            [a, b] = inputs
            c = a.minimum(b)
            return [b.minimum(a) if c is NotImplemented else c]

        if op_type in ["Relu"]:
            return [inputs[0].relu()]

        if op_type in ["Abs"]:
            return [inputs[0].abs()]

        if op_type in ["Log"]:
            return [inputs[0].log()]

        if op_type in ["Log1p"]:
            return [inputs[0].log1p()]

        if op_type in ["Exp"]:
            return [inputs[0].exp()]

        if op_type in ["Sigmoid"]:
            return [inputs[0].box_sigmoid()]

        if op_type in ["Softmax"]:
            return [inputs[0].box_softmax()]

        if op_type in ["Neg"]:
            return [- inputs[0]]

        raise NotImplementedError(
            ('the {} operation is not implemented by default in HZ for this ' +
             'version of parot').format(op_type))

    def promote_all(self):
        """
        Takes the largest components of the b vector and converts
        them to E columns. That is, `E` maps to `concat(E,diag(b))`
        and `b` maps to zero.

        Returns:
            HZ: resultant zonotope
        """
        c = self.c
        (B, *Cs) = c.shape
        HWC = reduce(lambda x, y: x * y, Cs, 1)
        b = self.b
        E = tf.reshape(b, [-1, HWC])
        E = tf.linalg.diag(E)
        E = tf.transpose(E, [2, 0, 1])
        E = tf.reshape(E, [-1, *Cs])
        E = tf.concat([self.E, E], axis=0)
        return HZ(c, tf.zeros_like(c), E)

    def to_box(self):
        """
        Return a box corresponding to the hybrid zonotope

        Returns:
            Box: a box
        """
        from .box import Box

        return Box(self.c, self.get_errors())

    @staticmethod
    def of_domain(self, domain):
        """
        Create an HZ using the `get_center_errors` method on the domain.

        Args:
            domain (Domain): domain to start from

        Returns:
            HZ: resultant zonotope
        """
        c, e = domain.get_center_errors()
        return HZ.of_ball_promoted(c, e)

    @staticmethod
    def of_minmax(x1, x2):
        """
        HZ from the difference between two tensors.

        Args:
            x1 (tf.Tensor): maximum/minimum tensor
            x2 (tf.Tensor): maximum/minimum tensor

        Returns:
            HZ: resultant zonotope
        """
        return HZ((x1 + x2) / 2, abs(x1 - x2) / 2, tf.zeros_like(x2))

    @staticmethod
    def of_E_columns(x, cols):
        """
        Given an input tensor, get an HZ instance with a computed E matrix.

        Args:
            x (tf.Tensor): Description
            cols (List[Int]): Description

        Returns:
            HZ: resultant zonotope
        """
        [B, *S] = x.shape
        E = tf.constant(cols)
        E = tf.expand_dims(E, axis=1)
        E = tf.tile(E, multiples=[1, B, *[1 for _ in S]])
        E = tf.reshape(E, [len(cols) * B, *S])
        return HZ(x, tf.zeros_like(x), E)
