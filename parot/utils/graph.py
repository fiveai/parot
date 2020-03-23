# Copyright 2020 FiveAI Ltd.
# All rights reserved.
#
# This file is part of the PaRoT toolbox, and is released under the
# "MIT License Agreement". Please see the LICENSE file that should
# have been included as part of this package.

from copy import deepcopy

import tensorflow as tf


def isin(z, xs): return any(z is x for x in xs)


def depends_on_any(z, xs):
    """
    Determine whether tensor `z` depends on `xs` in the dataflow graph.
    Reflexive, transitive.

    Args:
        z (tf.Tensor): input tensor
        xs (List[tf.Tensor]): list of tensors to test

    Returns:
        bool: True if there is a dependency, False otehrwise
    """
    # no variables allowed.
    assert all(isinstance(x, tf.Tensor) for x in xs + [z])

    if isin(z, xs):
        return True

    visited = set()
    front = [z]
    while len(front) != 0:
        y = front.pop()
        op = y.op
        parents = set(op.inputs)
        if any(isin(x, parents) for x in xs):
            return True
        parents -= visited
        front += [p for p in parents if not isin(p, front)]
        visited.add(y)
    return False


def depends_on(z, x): return depends_on_any(z, [x])


def get_between(zs, xs):
    """
    Given lists of Tensors `zs` and `xs` define the _graph between_ `zs` and
    `xs` as `G` to be the full subgraph of all vertices (`Tensor`s) `y`
    such that there exists `x \in xs` and `z \in zs` such that there is a path
    `x ~~ y` and a path `y ~~ z`.
    This method computes `G(zs,xs)` and provides a pair of dictionaries for
    traversing it.

    - `consumer_map` a map taking a tensor and returning the consumer ops
    of this tensor in `G`
    - `op_map` a map taking an op and returning a set of indices of the
    inputs of the op which are consumed by tensors which are in G.

    Args:
        zs (List[tf.Tensor]): input list of tensors
        xs (List[tf.Tensor]): output list of tensors

    Returns:
        Tuple[tf.Tensor]: returns `consumer_map` and `op_map`
    """

    consumer_map = {}
    op_map = {}
    visited = set()
    front = [z for z in zs]
    while len(front) != 0:
        y = front.pop()
        if y in visited:
            continue
        op = y.op
        parents = op.inputs
        for i, p in enumerate(parents):
            if depends_on_any(p, xs):
                if not isin(p, consumer_map):
                    consumer_map[p] = set()
                consumer_map[p].add(op)
                if not isin(op, op_map):
                    op_map[op] = set()
                op_map[op].add(i)
                if not isin(p, visited):
                    front.append(p)
        visited.add(y)

    return consumer_map, op_map


def clone_op(op, inputs, suffix="_", G=None):
    """
    Make a deep copy of `op` but change the name and optionally the graph
    that it belongs to. The name will be prependend with the context that the
    graph is currently in.

    Args:
        op (tf.Operation): tensorflow operation
        inputs (List[tf.Tensor]): list of inputs of the operation
        suffix (str, optional): suffix to the operation's name
        G (None, optional): tensorflow graph

    Returns:
        List[t.Tensor]: cloned operation outputs
    """
    G = G or tf.compat.v1.get_default_graph()
    nd = deepcopy(op.node_def)
    nd.name = G.unique_name(nd.name + suffix)
    cloned_op = tf.Operation(
        node_def=nd,
        g=G,
        inputs=inputs,
        original_op=op
    )

    for cloned_o, o in zip(cloned_op.outputs, op.outputs):
        if len(o.shape) == 0:
            cloned_o.set_shape([])
        else:
            [B, *C] = o.shape
            assert not (None in C)
            cloned_o.set_shape([None, *C])

    return cloned_op.outputs


def transform(*, outputs, inputs, transformed_inputs, transformer):
    """
    Creates an abstract transformer for the given input/output by finding
    all of the tensor ops between `input` and `output` and then transforming
    each of these according to the `transformer` function.
    If the transformer returns None for a particular output then that output is
    not transformed and the original tensor is used.
    Returns a list of transformed outputs with the same length as `outputs`.
    Will throw if an output is not dependent on any of the inputs.

    Will throw if an input or output is a `tf.Variable`.

    Does not traverse control_inputs, such as those attached when using
    `tf.control_dependencies`.

    Args:
        outputs (List[tf.Tensor]): list of output tensors
        inputs (List[tf.Tensor]): list of input tensors
        transformed_inputs (List[T]): list of transformed input tensors
        transformer (Callable[[tf.Operation, List[Union[T, tf.Tensor]]],
         List[Union[T, None]]]): transformer function

    Returns:
        List[T]: list of transformed outputs
    """
    assert len(inputs) is len(transformed_inputs)

    for z in outputs:
        # if some zs don't depend on the transformed version this will fail.
        assert depends_on_any(z, inputs)

    # `consumer_map` is a map from a tensor to ops that consume this tensor
    # and which are parents of `output`
    # `op_map` takes an op and returns the indices of the outputs of the op
    # which are parents of `output`.
    consumer_map, op_map = get_between(outputs, inputs)

    # `m` is a map taking a tensor to the abstraction domain item that is
    # representing that tensor.
    # Some tensors will map to `None`, this means that the tensor is on the
    # dependency graph between `input` and `output`, but the abstraction domain
    # item is not needed for this tensor.
    m = {input: transformed_input for (
        input, transformed_input) in zip(inputs, transformed_inputs)}

    # A set of tensors that must be explored before the algorithm terminates.
    front = [i for i in inputs]
    transformed_outputs = [None for o in outputs]

    # sometimes an output is exactly an input, in which case
    for i in inputs:
        for j, o in enumerate(outputs):
            if i is o:
                transformed_outputs[j] = m[i]

    while len(front) != 0:
        b = front.pop()
        # skip items that aren't dependencies for `output`
        if not isin(b, consumer_map):
            continue
        # get the consumers of b which are dependencies for `output`
        ops = consumer_map[b]
        for op in ops:
            if not isin(op, op_map):
                continue  # shouldn't happen
            omo = op_map[op]
            if not all(isin(op.inputs[i], m) for i in omo):
                # waiting for other producers, will be hit later.
                continue
            # Make a list of inputs to the transformed op. Look up each input
            # of `op` and if it has an abstract domain counterpart in `m` then
            # use that, otherwise just use the normal input.
            transformed_op_inputs = [((m[input] if m[input] is not None else
                                       input) if isin(i, omo) else input) for
                                     i, input in enumerate(op.inputs)]
            # check that at least one of the inputs in `inputs` is in the
            # abstract domain.
            transformed_op_outputs = transformer(op, transformed_op_inputs)
            for o, transformed_op_output in zip(op.outputs,
                                                transformed_op_outputs):
                for i, output in enumerate(outputs):
                    if o is output:
                        # transformer is allowed to return None, in which case
                        # just forward the original output.
                        transformed_outputs[i] = transformed_op_output if\
                            transformed_op_output is not None else output
                m[o] = transformed_op_output
                front.append(o)

    return transformed_outputs


def clone_subgraph(*, outputs, inputs, new_inputs, suffix="cloned"):
    """
    Take all of the tensorflow nodes between `outputs` and `inputs` and clone
    them but with `inputs` replaced with `new_inputs`.

    Args:
        outputs (List[tf.Tensor]): list of output tensors
        inputs (List[tf.Tensor]): list of input tensors
        new_inputs (List[tf.Tensor]): list of new input tensors
        suffix (str, optional): suffix to the transformed operation names

    Returns:
        List[T]: list of transformed outputs
    """
    return transform(outputs=outputs, inputs=inputs,
                     transformed_inputs=new_inputs,  transformer=lambda op,
                     inputs: clone_op(op, inputs, suffix=suffix))
