# Copyright 2020 FiveAI Ltd.
# All rights reserved.
#
# This file is part of the PaRoT toolbox, and is released under the
# "MIT License Agreement". Please see the LICENSE file that should
# have been included as part of this package.

"""
Experimental setup used for the PaRoT DiffAI comparison
"""

import warnings
import os
import json
import time
import argparse

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from parot.domains import Box, HZ
from parot.utils.testing import PGD
from parot.properties import Ball, BallDemoted, BallPromoted, Fourier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

model_ids = ["FFNN", "ConvSmall", "ConvMed", "ConvBig", "ConvSuper", "Skip"]
domain_ids = ['box', 'hz']
property_ids = ['ball', 'ball_demoted', 'ball_promoted', 'fourier']
dataset_ids = ['MNIST', 'CIFAR10']


def DiffAIModels(id, input_shape, num_classes):
    """
    Factory for different models identified by `id` with
    inputs of `input_shape` and an ouput of `num_classes`

    Args:
        id (str): string description of the model
        input_shape (np.ndarray): shape of the input
        num_classes (int): number of output classes

    Returns:
        tf.keras.model.Sequential: desired model
    """

    if id == 'FFNN':
        return tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(
                num_classes, activation='softmax', name='y_pred')
        ])
    elif id == 'ConvSmall':
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                16, kernel_size=(4, 4), strides=2,
                activation='relu',
                input_shape=input_shape),
            tf.keras.layers.Conv2D(32, (4, 4), strides=2, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(
                num_classes, activation='softmax', name='y_pred')
        ], name='ConvSmall')

    elif id == 'ConvMed':
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                16, kernel_size=(4, 4), strides=2,
                activation='relu', padding='same',
                input_shape=input_shape),
            tf.keras.layers.Conv2D(32, (4, 4), strides=2,  padding='same',
                                   activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(
                num_classes, activation='softmax', name='y_pred')
        ], name='ConvMed')

    elif id == 'ConvBig':
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), strides=1,
                activation='relu', padding='same',
                input_shape=input_shape),
            tf.keras.layers.Conv2D(32, (4, 4), strides=2,  padding='same',
                                   activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), strides=1,  padding='same',
                                   activation='relu'),
            tf.keras.layers.Conv2D(64, (4, 4), strides=2,  padding='same',
                                   activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(
                num_classes, activation='softmax', name='y_pred')
        ], name='ConvBig')

    elif id == 'ConvSuper':
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), strides=1,
                activation='relu', padding='valid',
                input_shape=input_shape),
            tf.keras.layers.Conv2D(32, (4, 4), strides=1,  padding='valid',
                                   activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), strides=1,  padding='valid',
                                   activation='relu'),
            tf.keras.layers.Conv2D(64, (4, 4), strides=1,  padding='valid',
                                   activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(
                num_classes, activation='softmax', name='y_pred')
        ], name='ConvSuper')

    elif id == 'Skip':
        input_ = tf.keras.layers.Input(shape=input_shape)
        m1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                16, kernel_size=(3, 3), strides=1,
                activation='relu', padding='valid',
                input_shape=input_shape),
            tf.keras.layers.Conv2D(16, (3, 3), strides=1,  padding='valid',
                                   activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), strides=1,  padding='valid',
                                   activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(200)
        ])
        m2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                32, kernel_size=(4, 4), strides=1,
                activation='relu', padding='valid',
                input_shape=input_shape),
            tf.keras.layers.Conv2D(32, (4, 4), strides=1,  padding='valid',
                                   activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(200)
        ])
        o1 = m1(input_)
        o2 = m2(input_)
        merged = tf.keras.layers.concatenate([o1, o2])

        output = tf.keras.layers.ReLU()(merged)
        output = tf.keras.layers.Dense(200, activation='relu')(output)
        output = tf.keras.layers.Dense(
            num_classes, activation='softmax')(output)
        return tf.keras.models.Model(inputs=[input_], outputs=output,
                                     name='y_pred')

    raise ValueError('model id "%s" not available' % id)


def loss_fn(y_true, y_pred):
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))


def setup_graph(model_id, training_iterator, input_shape, num_classes,
                learning_rate, pgd_step_count, pgd_learning_rate,
                batch_size, domain_max, domain_min,
                prop=Ball, domain=Box
                ):
    """
    Setups up the tensorflow graph for a given `model_id` with an
    input of `input_shape` and an output of `num_classes`,
    assuming a training dataset represented by a `training_iterator`

    Args:
        model_id (str): string identifier of the model to be used
        training_iterator (Iterator): iterator for the training
            dataset
        input_shape (ndarray): shape of the input of the model
        num_classes (int): number of output classes in the model
        learning_rate (float): Optimiser learning rate.
        pgd_step_count: number of steps to run the PGD optimiser for
        pgd_learning_rate: learning rate for PGD optimiser.
        batch_size (int): size of the batch
        domain_max (float): maximum value that the input tensor is allowed to
            take
        domain_min (float): minimum value that the input tensor is allowed to
            take
        prop (function(x,eps) -> domain): the type of property that it
            should be trained against.

    Returns:
        training_ops (dict): dictionary containg the training
            operations to be used in the training of the model
        testing_ops (dict): dictionary containg the testing
            operations to be used in the testing of the model
        placeholder_vars (dict): dictionary containing the
            placeholder variables that must be substituted
            before running the model
    """
    # eps and lam are scalar placeholder vars
    eps = tf.compat.v1.placeholder(tf.float32, shape=(), name='eps')
    adversary_eps = tf.compat.v1.placeholder(tf.float32, shape=(),
                                             name="adversary_eps")
    lam = tf.compat.v1.placeholder(tf.float32, shape=(), name='lam')

    # get the model
    model = DiffAIModels(model_id, input_shape, num_classes)
    optim = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    # training operations
    x, y_true = training_iterator.get_next()
    x = tf.reshape(x, shape=[batch_size, *input_shape])

    y_pred = model(x)

    regular_loss = loss_fn(y_true, y_pred)

    x_box = prop(eps).of(domain, x)

    [y_pred_box] = x_box.transform(outputs=[y_pred], input=x)
    y_adversary = y_pred_box.get_center() - y_pred_box.get_errors() * (
        tf.cast(tf.one_hot(y_true, depth=num_classes), tf.float32) * 2 - 1)
    adversary_loss = loss_fn(y_true,  y_adversary)

    trainable_vars = tf.compat.v1.trainable_variables()
    regularisation_loss = tf.add_n(
        [tf.compat.v1.nn.l2_loss(v) for v in trainable_vars],
        name='regularization_loss')

    combined_loss = lam * adversary_loss + regular_loss +\
        0.01 * regularisation_loss

    # training operation
    train_op = optim.minimize(
        combined_loss, var_list=model.variables,
        global_step=tf.compat.v1.train.get_global_step())

    training_ops = {
        'train': train_op
    }

    # testing operations
    x_t = tf.compat.v1.placeholder(
        tf.float32, shape=[None, *x_test.shape[1:]], name='x_t')
    y_t = tf.compat.v1.placeholder(
        tf.float32, shape=[None, *y_test.shape[1:]], name='y_t')

    # compute accuracy as the number of correctly predicted classes
    y_pred_ = model(x_t)

    y_pred_int = tf.cast(tf.argmax(y_pred_, axis=1), tf.uint8)
    y_test_int = tf.cast(y_t, tf.uint8)

    compare_op_test = tf.cast(tf.equal(y_pred_int, y_test_int), tf.float32)
    test_op = tf.stop_gradient(tf.reduce_mean(compare_op_test), name='test_op')

    x_t_elem = tf.compat.v1.placeholder(
        tf.float32, shape=(1, *x_test.shape[1:]), name='x_t_elem')
    y_t_elem = tf.compat.v1.placeholder(
        tf.float32, shape=(1, *y_test.shape[1:]), name='y_t_elem')
    y_t_elem_int = tf.cast(y_t_elem, tf.uint8)
    y_pred_elem = model(x_t_elem)

    # Compute the transformer on x_t_elem
    x_box_ = prop(eps).of(domain, x_t_elem)
    [y_pred_box_] = x_box_.transform(
        outputs=[y_pred_elem],
        input=x_t_elem
    )
    y_test_one_hot = tf.cast(tf.one_hot(y_t_elem_int, num_classes), tf.float32)
    (y_pred_box_center, y_pred_box_error) = y_pred_box_.get_center_errors()

    y_verify_adversary = y_pred_box_center - y_pred_box_error * \
        (y_test_one_hot * 2 - 1)
    y_verify_adversary_int = tf.cast(
        tf.argmax(y_verify_adversary, axis=1), tf.uint8)
    verify_adversary_test_op = 1 - \
        tf.reduce_mean(tf.cast(tf.equal(y_verify_adversary_int, y_t_elem_int),
                               tf.float32), name='verify_adversary_test_op')

    # compute an adversarial example.
    pgd = PGD(
        property=prop,
        domain=domain,
        epsilon=adversary_eps,
        learning_rate=pgd_learning_rate,
        step_count=pgd_step_count,
        domain_max=domain_max,
        domain_min=domain_min,
    )
    # `x_adversary` is a perturbed input, `y_adversary` is the perturbed output
    x_pgd_adversary, [y_pgd_adversary_], _ = pgd(
        x_t_elem, [y_pred_elem], loss_fn(y_t_elem, y_pred_elem))

    y_pgd_adversary_int = tf.cast(
        tf.argmax(y_pgd_adversary_, axis=1), tf.uint8)
    pgd_adversary_test_op =\
        1 - tf.reduce_mean(
            tf.cast(tf.equal(y_pgd_adversary_int, y_t_elem_int), tf.float32),
            name='pgd_adversary_test_op')

    testing_ops = {
        'test': test_op,
        'verify_adversary_test': verify_adversary_test_op,
        'pgd_adversary_test': pgd_adversary_test_op,
        'y_verify_adversary_int': y_verify_adversary_int,
        'y_pgd_adversary_int': y_pgd_adversary_int,
        'y_t_elem_int': y_t_elem_int
    }

    # placeholder variables
    placeholder_vars = {
        'eps': eps,
        'adversary_eps': adversary_eps,
        'lam': lam,
        'x_t': x_t,
        'y_t': y_t,
        'x_t_elem': x_t_elem,
        'y_t_elem': y_t_elem
    }

    return training_ops, testing_ops, placeholder_vars


def train(ops, placeholders, iterator, name, max_epochs, n_runs, eps, lam,
          parent_folder='checkpoints', testing_ops={}, testing_data={}
          ):
    """
    Run the training multiple times and save each checkpoint
    (0, ..., `n_runs`-1) to a folder named `name` inside `parent_folder`.

    Args:
        ops (dict): dictionary of instances of training ops to be run
        placeholders (dict): dictionary with the placeholder variables
            to be replaced in the session
        iterator (Iterator): training dataset iterator
        name (str): folder inside `parent_folder` where the training
            configuration and checkpoints will be stored
        max_epochs (int): maximum number of epochs to run training for
        n_runs (int): number of runs to perform at this training stage
        eps (float): epsilon value for the box training
        lam (float): lambda value in the combined loss function
        parent_folder (str): path to the folder where this training
            session will be stored
        testing_ops (dict, optional): if passed, testing accuracy will be
            computed for debugging purposes
        testing_data (dict, optional): if testing_ops is passed,
            required for computing accuracy
    """
    # create the folder with the name of the training op
    os.system(
        'mkdir -p %s > /dev/null 2>&1' % os.path.join(parent_folder, name))

    # write the config
    config = {
        'eps': eps,
        'lam': lam
    }
    with open(os.path.join(parent_folder, name, 'config.json'), 'w') as f:
        json.dump(config, f)

    # substitute the placeholder operations
    sub_placeholders = {
        placeholders['eps']: eps,
        placeholders['lam']: lam
    }

    for r in range(n_runs):
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            i = 0
            print('Run %d of %d for training "%s"...' % (r + 1, n_runs, name))

            t_cum = 0
            for epoch in range(max_epochs):
                t_start_epoch = time.time()

                sess.run(iterator.initializer)
                while True:
                    i += 1
                    try:
                        sess.run(ops['train'], feed_dict=sub_placeholders)
                    except tf.errors.OutOfRangeError:
                        break

                delta_t = time.time() - t_start_epoch
                t_cum += delta_t

                test_acc = 'N/A'
                if len(testing_ops) != 0 and len(testing_data) != 0:
                    test_acc = sess.run(
                        testing_ops['test'],
                        feed_dict={
                            placeholders['eps']: eps,
                            placeholders['lam']: lam,
                            placeholders['x_t']: testing_data['x_t'],
                            placeholders['y_t']: testing_data['y_t']})

                print(
                    ("Epoch: %d, accuracy: %s, elapsed time (s): %.3f," +
                        " cumulative time (s): %.3f") %
                    (epoch, test_acc, delta_t, t_cum))

            # add checkpoint and return
            saver = tf.compat.v1.train.Saver()
            saver.save(
                sess, os.path.join(parent_folder, name, str(r) + '.ckpt'))


def test_folder(checkpoint_folder, ops, placeholders, x_test, y_test, dataset,
                test_epsilon, pgd_sample_size):
    """
    Given a `checkpoint_folder` with at least one checkpoint inside, run
    the testing operations in `ops` using the placeholder variables in
    `placeholders`

    Args:
        checkpoint_folder (str): path to a folder with checkpoints for a
            certain model; it should also contain a 'config.json' with the
            parameters used in the training stage
        ops (dict): dictionary of testing ops to be run
        placeholders (dict): dictionary with the placeholder variables
            to be replaced in the session
        x_test (ndarray): input testing data
        y_test (ndarray): output testing data
        dataset (TYPE): Description
        test_epsilon (TYPE): Description
        pgd_sample_size (int): number of samples to be used in PGD; must be
            at most the number of points in the testing dataset

    Returns:
        ops_results (list): list of results of the testing operations
    """
    models = [
        f
        for f in os.listdir(checkpoint_folder)
        if os.path.isdir(os.path.join(checkpoint_folder, f)) and dataset in f]

    sub_adv_vars = {
        placeholders['x_t']: x_test,
        placeholders['y_t']: y_test
    }

    np.random.seed(0)
    adversary_sample_indices = np.random.choice(
        x_test.shape[0], pgd_sample_size, replace=False)

    # test only on these indices
    sub_test_vars = {
        placeholders['x_t']: x_test[adversary_sample_indices],
        placeholders['y_t']: y_test[adversary_sample_indices]
    }

    results = {}
    for m in models:
        model_folder = os.path.join(checkpoint_folder, m)
        ckpts = [
            f[:-10]
            for f in os.listdir(model_folder)
            if (os.path.isfile(os.path.join(model_folder, f)) and
                '.ckpt.meta' in f)]

        # read the config of this model (all checkpoints will share it)
        with open(os.path.join(model_folder, 'config.json'), 'r') as f:
            config = json.load(f)

        sub_test_vars[placeholders['eps']] = test_epsilon
        sub_test_vars[placeholders['lam']] = config['lam']

        sub_adv_vars[placeholders['eps']] = test_epsilon
        sub_adv_vars[placeholders['lam']] = config['lam']
        sub_adv_vars[placeholders['adversary_eps']] = test_epsilon

        acc = []
        verify = []
        pgd = []
        for ckpt in ckpts:
            with tf.compat.v1.Session() as sess:
                saver = tf.compat.v1.train.Saver()
                saver.restore(sess, os.path.join(model_folder, ckpt + '.ckpt'))

                acc.append(
                    float(sess.run(ops['test'], feed_dict=sub_test_vars)))

                pgd_result = 0
                verify_result = 0
                for idx in tqdm(adversary_sample_indices):
                    sub_adv_vars[placeholders['x_t_elem']] = [x_test[idx]]
                    sub_adv_vars[placeholders['y_t_elem']] = [y_test[idx]]
                    output = sess.run(
                        [ops['verify_adversary_test'],
                         ops['pgd_adversary_test'],
                         ops['y_verify_adversary_int'],
                         ops['y_pgd_adversary_int'],
                         ops['y_t_elem_int']], feed_dict=sub_adv_vars)
                    [verify_ad_test, pgd_ad_test, y_verify_adversary_int,
                     y_pgd_adversary_int, y_t_elem_int] = output

                    verify_result += verify_ad_test
                    pgd_result += pgd_ad_test

                verify.append(verify_result / pgd_sample_size)
                pgd.append(pgd_result / pgd_sample_size)

                print('%s: test error %.2f, PGD %.2f, verify %.2f' %
                      (m,
                       100 * (1 - acc[-1]),
                       100 * (pgd[-1]),
                       100 * (verify[-1])))

        results[m] = {
            'name': m,
            'epsilon': float(config['eps']),
            'lam': float(config['lam']),
            'test_epsilon': float(test_epsilon),
            'acc': acc,
            'verify': verify,
            'pgd': pgd
        }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DiffAI comparison experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', type=str, required=True, choices=model_ids,
        help='Define the model to use')
    parser.add_argument(
        '--domain', choices=domain_ids, help="specify the domain type",
        required=True)
    parser.add_argument(
        '--property', choices=property_ids, help="specify the property type",
        required=True)
    parser.add_argument(
        '--dataset', '-D', choices=dataset_ids,
        help="specify the dataset to be used", required=True)

    parser.add_argument(
        '--buffer-size', type=int, default=5000, help='dataset buffer size')
    parser.add_argument(
        '--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument(
        '--test-only', dest='test', action="store_true",
        help="simply test the models found in checkpoints only")
    parser.set_defaults(test=False)
    parser.add_argument(
        '-e', '--epsilon', nargs="+", type=float, default=[0.1],
        help="the width of the property")
    parser.add_argument(
        '-l', '--lam', nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument(
        '--epochs', type=int, default=200,
        help="number of epochs to train for")
    parser.add_argument(
        '--learning-rate', type=float, default=0.0001,
        help="learning rate of training optimiser")
    parser.add_argument(
        '--runs', type=int, default=1,
        help="number of times to train the model. ")
    parser.add_argument(
        '--pgd-step-count', type=int, default=100,
        help="number of steps to run pgd optimiser for")
    parser.add_argument(
        '--pgd-learning-rate', type=float, default=1.0,
        help="learning rate of pgd optimiser")
    parser.add_argument(
        '--test-verify-sample-size', type=int, default=500,
        help="number of samples to get PGD and verify bounds on test data.")
    parser.add_argument(
        '--test-epsilon', type=float, default=0.1,
        help="epsilon to test against")
    args = parser.parse_args()

    # Data configuration
    if args.dataset == 'MNIST':
        (x_train, y_train), (x_test, y_test) =\
            tf.keras.datasets.mnist.load_data()
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
    elif args.dataset == 'CIFAR10':
        (x_train, y_train), (x_test, y_test) =\
            tf.keras.datasets.cifar10.load_data()

    # Prepare datasets
    x_train = (x_train / 255.0).astype(np.float32)
    x_test = (x_test / 255.0).astype(np.float32)

    # depending on the dataset, a data reshape might be required
    if len(y_train.shape) > 1:
        # reshape the vector
        y_train = y_train.reshape(y_train.shape[0])
        y_test = y_test.reshape(y_test.shape[0])

    # normalize based on the dataset
    if args.dataset == 'MNIST':
        def norm(x):
            return (x - 0.1307) / 0.3081

        x_train = norm(x_train)
        x_test = norm(x_test)
        domain_min = norm(0.0)
        domain_max = norm(1.0)
    elif args.dataset == 'CIFAR10':
        def norm(x):
            x[..., 0] = (x[..., 0] - 0.4914) / 0.2023
            x[..., 1] = (x[..., 1] - 0.4822) / 0.1994
            x[..., 2] = (x[..., 2] - 0.4465) / 0.2010
            return x

        x_train = norm(x_train)
        x_test = norm(x_test)
        domain_min = norm(np.zeros([3]))
        domain_max = norm(np.ones([3]))

    # image inputs: it's either a 2D (no RGB) or 3D vector
    input_shape = [x_train.shape[1], x_train.shape[2], x_train.shape[3]]

    num_classes = len(set(list(y_train) + list(y_test)))
    batch_size = args.batch_size

    model_id = args.model
    property_id = args.property
    if property_id == "ball":
        prop = Ball
    elif property_id == "ball_promoted":
        prop = BallPromoted
    elif property_id == "ball_demoted":
        prop = BallDemoted
    elif property_id == "fourier":
        prop = Fourier

    domain_id = args.domain
    if domain_id == 'box':
        domain = Box
    elif domain_id == 'hz':
        domain = HZ

    dataset_train = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(
            buffer_size=args.buffer_size).batch(
            batch_size, drop_remainder=True)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset_train)

    # Set up the graph and collect the operations and placeholder variables
    training_ops, testing_ops, placeholders = setup_graph(
        model_id,
        iterator,
        input_shape,
        num_classes,
        learning_rate=args.learning_rate,
        pgd_step_count=args.pgd_step_count,
        pgd_learning_rate=args.pgd_learning_rate,
        batch_size=batch_size,
        prop=prop,
        domain=domain,
        domain_max=domain_max,
        domain_min=domain_min,
    )

    # Training under different conditions
    n_runs = args.runs

    eps = args.epsilon
    lam = args.lam
    if not args.test:
        for l in lam:
            if l == 0.0:
                train(
                    training_ops,
                    placeholders,
                    iterator,
                    name=args.dataset + "-" + model_id + "-" + "regular",
                    lam=l,
                    eps=0.1,
                    n_runs=n_runs,
                    max_epochs=args.epochs,
                    testing_ops=testing_ops,
                    testing_data={'x_t': x_test, 'y_t': y_test},
                    parent_folder=os.path.join('checkpoints', model_id)
                )
            else:
                for e in eps:
                    name = args.dataset +\
                        "-" + model_id +\
                        "-combined-" + property_id +\
                        "-" + str(e).replace('.', '_') +\
                        "-" + str(l).replace('.', '_')
                    train(
                        training_ops,
                        placeholders,
                        iterator,
                        name,
                        eps=e, lam=l,
                        n_runs=n_runs,
                        max_epochs=args.epochs,
                        testing_ops=testing_ops,
                        testing_data={'x_t': x_test, 'y_t': y_test},
                        parent_folder=os.path.join('checkpoints', model_id)
                    )

    # Testing outcome
    results = test_folder(
        os.path.join(os.getcwd(), 'checkpoints', model_id),
        testing_ops,
        placeholders,
        x_test, y_test,
        args.dataset,
        pgd_sample_size=args.test_verify_sample_size,
        test_epsilon=args.test_epsilon
    )

    # Save the raw outcome to a file
    with open('%s_%s.json' % (args.dataset, model_id), 'w') as f:
        json.dump(results, f, indent=2)
