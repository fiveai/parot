# Copyright 2020 FiveAI Ltd.
# All rights reserved.
#
# This file is part of the PaRoT toolbox, and is released under the
# "MIT License Agreement". Please see the LICENSE file that should
# have been included as part of this package.

from abc import abstractmethod

import tensorflow as tf

from parot.utils import graph


class Domain():
    """
    An abstraction domain is a functor mapping tensors to items in a domain
    set and mapping tensor ops to _abstract transformers_.
    The idea is that you use the domain to represent a set of points and then
     feed the entire set through the network at once to get guarantees on the
     image set.

    To implement an abstraction domain, provide the following abstract methods:
    - `get_center`
    - `get_errors`
    - `get_parameters`
    - `evaluate`
    - The static method `transform_op`
    """
    @abstractmethod
    def get_errors(self):
        """
        Returns a tensor representing the halfwidths of the axis-aligned
        bounding box containing all of the points in the domain.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_center(self):
        """
        Returns the center point of the axis-aligned bounding box containing
        all of the points in the domain.
        Description
        """
        raise NotImplementedError()

    def get_center_errors(self):
        """
        Alias for `(self.get_center(), self.get_errors())`.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: tuple of center and errors
        """
        return (self.get_center(), self.get_errors())

    def get_adversary(self, y_true):
        """
        Draws an axis-aligned box around the domain element and then finds the
        point in this box which is furthest from the tensor 'true'

        Args:
            y_true (tf.Tensor): the true output

        Returns:
            tf.Tensor: the furthest tensor from `y_true` in the box
        """
        c = self.get_center()
        s = self.get_errors()
        g = tf.sign(c - y_true)
        return c + s * g

    def is_inside_bounding_box(self, x):
        """
        Determines if x is inside the abstraction domain

        Args:
            x (tf.Tensor)

        Returns:
            bool: True if x is inside the domain
        """
        c, e = self.get_center_errors()
        return tf.reduce_all(abs(x - c) <= e)

    @staticmethod
    @abstractmethod
    def transform_op(op, inputs):
        """
        Function that is called whenever the `transform` method needs to
        transform an op.
        Takes an operation `op` and a list of inputs which can be either a
        tensor or an instance of the class `transform_op` is implemented for.
        The length of `inputs` is the same as `op.inputs`.
        The function should return a list of outputs with the same length as
        `op.outputs`.

        Each item in the list can be:
            - An instance of the domain class.
            - `None` which means that the output should use corresponding
             `op.outputs` value.
            - A tensor.

        Args:
            op (tf.Operation): the operation to transform
            inputs (List[tf.Tensor]): list of inputs
        """
        raise NotImplementedError()

    def transform(self, *, outputs, input, post=None):
        """
        Creates an abstract transformer for the given input/output by finding
         all of the tensor ops between `input` and `output` and then
         transforming each of these to the abstraction domain.

        Args:
            outputs (List[tf.Tensor]): list of output tensors
            input (tf.Tensor): input tensor
            post (Callable, optional): function that is called for each op
             explored by the algorithm and can be used to attach debug
             summaries. Arguments to `post` are the original tensor and the
             new AbstractDomain object derived from it.

        Returns:
            List[tf.Tensor]: list of transformed outputs
        """
        def transformer(op, inputs):
            if any(isinstance(i, self.__class__) for i in inputs):
                # use the static transform_op method on the class.
                outputs_abstract = self.__class__.transform_op(op, inputs)
                # Remember to return a list!
                assert isinstance(outputs_abstract, list)
                if outputs_abstract == NotImplemented:
                    raise NotImplementedError("Transformer of " + op.type +
                                              " is not implemented for " + str(
                                                  self.__class__))
            # if there are no inputs in the abstract domain then we don't need
            # them in the output domain either.
            else:
                outputs_abstract = [None for o in op.outputs]
            for o, o_abs in zip(op.outputs, outputs_abstract):
                if (o_abs is not None) and (post is not None):
                    post(o, o_abs)
            return outputs_abstract

        transformed_outputs = graph.transform(
            outputs=outputs, inputs=[input], transformer=transformer,
            transformed_inputs=[self])
        return transformed_outputs
