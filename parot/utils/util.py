# Copyright 2020 FiveAI Ltd.
# All rights reserved.
#
# This file is part of the PaRoT toolbox, and is released under the
# "MIT License Agreement". Please see the LICENSE file that should
# have been included as part of this package.

import io
import math

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm


def linterp(x1, x2, y1, y2, x):
    """
    Linearly interpolate (x1, y1) and (x2, y2) at x

    Args:
        x1 (tf.Tensor): x of the first point
        x2 (tf.Tensor): x of the second point
        y1 (tf.Tensor): y of the first point
        y2 (tf.Tensor): y of the second point
        x (tf.Tensor): points at which the y is desired

    Returns:
        tf.Tensor: y at each of the points in x
    """
    return tf.div_no_nan((x * (y2 - y1) + y1 * x2 - x1 * y2), (x2 - x1))


def to_log_heatmap(image, lower=-10, upper=10, cm_name="RdYlBu"):
    """
    Take logs and map to a colour from a heatmap. Clamps values outside
     `lower` and `upper` parameters.

    Args:
        image (tf.Tensor): input image
        lower (float, optional): lower bound
        upper (float, optional): upper bound
        cm_name (str, optional): colourmap name

    Returns:
        tf.Tensor: resulting image image
    """
    image = tf.log(image) / math.log(10)
    image = tf.where(tf.is_nan(image), lower * tf.ones_like(image), image)
    image = tf.clip_by_value(image, lower, upper)
    image = linterp(lower, upper, 0, 255, image)
    image = tf.cast(image, tf.int32)
    image = tf.clip_by_value(image, 0, 255)

    # define colormap and apply it to the image
    cmap = cm.get_cmap(cm_name)
    cmap = [cmap(i)[:3] for i in range(0, 256)]
    cmap = tf.constant(cmap, dtype=tf.float32)
    image = tf.gather(cmap, image)
    return image


def gridify(input, pad=2, I=None, J=None):
    """
    Takes as input a tensor of shape `(H, W, C, CO)` and returns a tensor of
    shape `(1, (H+2*pad) * I, (W+2*pad) * J, CO)`.
    Useful to visualise activation layers of images.
    If the input tensor has rank 3 it is extended so that `CO=1`.
    If `I` and `J` are not given they are calculated as the integer
     factorisation `C = I * J` closest to a square.

    Args:
        input (tf.Tensor): input tensor with shape (H, W, C, CO)
        pad (int, optional): padding
        I (None, optional): image height scale
        J (None, optional): image width scale

    Returns:
        tf.Tensor: resultant gridified tensor
    """
    if len(input.get_shape()) == 3:
        input = tf.expand_dims(input, axis=-1)
    x = tf.pad(input, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]))
    H, W, C, CO = input.get_shape()
    H = H + 2 * pad
    W = W + 2 * pad
    if J is None:
        J = int(math.sqrt(int(C)))
        while (C % J != 0):
            J -= 1
        I = C // J
    assert I * J == C
    x = tf.transpose(x, (2, 0, 1, 3))
    x = tf.reshape(x, (J, I * H, W, CO))
    x = tf.transpose(x, (0, 2, 1, 3))
    x = tf.reshape(x, (1, J * W, I * H, CO))
    x = tf.transpose(x, (0, 2, 1, 3))
    return x


def imagify(input, log_scale=True):
    """
    Given an input tensor, return an image that represents it.

    Args:
        input (tf.Tensor): input tensor
        log_scale (bool, optional): whether to use log scaling

    Returns:
        tf.Tensor: image tensor
    """
    sh = list(input.shape)
    if len(sh) == 4:
        # it's a batch of images
        if (sh[3] == 3):
            # it's a batch of colour images
            return input[:1, :, :, :]
        else:
            # show the channels as a grid.
            input = input[0, :, :, :]
            if log_scale:
                input = to_log_heatmap(input)
            else:
                input = tf.expand_dims(input, axis=-1)
            input = gridify(input)
            return input
    elif len(sh) == 5 and (sh[4] == 3 or sh[4] == 1):
        # it's a batch of many-channelled rgb or scalar images. Convert to a
        # grid of RGB or scalar images.
        input = input[0, :, :, :, :]
        input = gridify(input)
        return input
    elif len(sh) == 3:
        # batched 2D no channels
        input = input[0, :, :]
        input = tf.expand_dims(input, axis=0)
        if log_scale:
            input = to_log_heatmap(input)
        else:
            input = tf.expand_dims(input, axis=-1)
        return input
    elif len(sh) == 2:
        # batched linear;; gridify to make it look like an image.
        input = input[0, :]
        input = tf.expand_dims(input, axis=0)
        input = tf.expand_dims(input, axis=0)
        if log_scale:
            input = to_log_heatmap(input)  # [1,1,N,3]
        else:
            input = tf.expand_dims(input, axis=-1)  # [1,1,N,1]
        input = gridify(input, pad=0)
        return input

    else:
        return NotImplemented


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.

    Args:
        figure: matplotlib plot

    Returns:
        tf.image: resultant image
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_with_name(image, class_name):
    """
    Given an input tensor and its class name, return an image of it.

    Args:
        image (tf.Tensor): image tensor
        class_name (tf.Tensor): tensor with the class

    Returns:
        tf.Tensor: output image
    """
    return image_grid([{image, class_name}], 1)


def image_grid(items, height=3):
    """Return a height x height grid of the items as a matplotlib figure.

    Args:
        items (List): list of image inputs
        height (int, optional): number of image tiles in a column

    Returns:
        Matplotlib figure
    """
    # Create a figure to contain the plot.
    width = math.ceil(len(items) / height)

    figure = plt.figure(figsize=(2 * height, 2 * width))
    i = 0
    for item in items:
        # Start next subplot.
        plt.subplot(height, height, i + 1, title=item.class_name)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(item.image, cmap="RdYlBu")
        i += 1

    return figure
