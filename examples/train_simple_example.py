# Copyright 2020 FiveAI Ltd.
# All rights reserved.
#
# This file is part of the PaRoT toolbox, and is released under the
# "MIT License Agreement". Please see the LICENSE file that should
# have been included as part of this package.
"""
Example to showcase the ease of use of PaRoT
"""

import warnings
import os

import tensorflow as tf
import numpy as np

from parot.domains import Box
from parot.properties import Ball

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


def MNISTNorm(x):
    return (x - 0.1307) / 0.3081


def loss_fn(y_true, y_pred):
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))


# load the MNIST data
(x_train, y_train), (x_test, y_test) =\
    tf.keras.datasets.mnist.load_data()
x_train = (x_train[..., tf.newaxis] / 255.0).astype(np.float32)
x_test = (x_test[..., tf.newaxis] / 255.0).astype(np.float32)

x_train = MNISTNorm(x_train)
x_test = MNISTNorm(x_test)

input_shape = [28, 28, 1]
num_classes = 10

# get a dataset for training by slicing it
dataset_train = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(buffer_size=5000).batch(32)
iterator = tf.compat.v1.data.make_initializable_iterator(dataset_train)
x, y_true = iterator.get_next()

# define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
y_pred = model(x)

# ADVERSARY DEFINITION (change required for robust training)
x_box = Ball(0.1).of(Box, x)
[y_box] = x_box.transform(outputs=[y_pred], input=x)
y_adversary = y_box.get_adversary(y_true, num_classes)

# combined loss function
regular_loss = loss_fn(y_true, y_pred)
adversary_loss = loss_fn(y_true, y_adversary)
combined_loss = regular_loss + 0.1 * adversary_loss

# define the optimizer and the training operation
optim = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_op = optim.minimize(combined_loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(1, 201):
        sess.run(iterator.initializer)

        while True:
            try:
                sess.run(train_op)
            except tf.errors.OutOfRangeError:
                break

        print("Epoch: %d of 200" % epoch)

    # add checkpoint and return
    print('Saving checkpoint...')
    saver = tf.compat.v1.train.Saver()
    saver.save(
        sess,
        os.path.join(os.getcwd(), 'mnist_example.ckpt'))
