"""
Copyright (C) 2018 Shane Steinert-Threlkeld

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""

from __future__ import print_function, division
import itertools
import os

import numpy as np
import tensorflow as tf
import pandas as pd
import psutil
from tqdm import tqdm

import partition


def run_trial(params, out_dir):

    # model input and output sizes
    input_size = params['points'].shape[1]
    output_size = params['num_labels']
    column_name = 'point'

    the_partition = partition.Partition(params['points'], params['num_labels'],
                                        np.zeros(input_size), params['temp'],
                                        params['conv'])
    part = the_partition.partition
    points = the_partition.points

    train_split = 0.75
    train_bins = {label: part[label][:int(train_split*len(part[label]))] for
                  label in part}
    test_bins = {label: part[label][int(train_split*len(part[label])):] for
                 label in part}

    # oversample from all but biggest bins
    # note: partition.labelled_pts will no longer be true labels for training
    # data, only for test
    max_train_bin = max(len(train_bins[label]) for label in train_bins)
    train_bins = {label: np.random.choice(train_bins[label], max_train_bin,
                                          replace=True)
                  if 0 < len(train_bins[label]) < max_train_bin else
                  train_bins[label] for label in train_bins}

    def from_bins_to_xy(bins):
        x = np.vstack([points[bins[label]] for label in bins])
        # TODO: why does this sometimes return floats, and therefore require
        # the astype(int)?
        y = np.concatenate([[label]*len(bins[label]) for label in
                            bins]).astype(int)
        return x, y

    train_x, train_y = from_bins_to_xy(train_bins)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={column_name: train_x},
        y=train_y,
        batch_size=32,
        num_epochs=params['num_epochs'],
        shuffle=True)

    test_x, test_y = from_bins_to_xy(test_bins)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={column_name: test_x},
        y=test_y,
        num_epochs=1,
        batch_size=len(test_x),
        shuffle=False)

    total_x, total_y = from_bins_to_xy(part)
    total_input_train = tf.estimator.inputs.numpy_input_fn(
        x={column_name: total_x},
        y=total_y,
        batch_size=32,
        num_epochs=params['num_epochs'],
        shuffle=True)
    total_input_test = tf.estimator.inputs.numpy_input_fn(
        x={column_name: total_x},
        y=total_y,
        batch_size=len(total_x),
        num_epochs=1,
        shuffle=False)

    # main network training
    network_params = {
        'hidden_units': [32, 32],
        'activation': tf.nn.relu,
        'optimizer': tf.train.AdamOptimizer(),
        'model_dir': out_dir
    }

    point_column = tf.feature_column.numeric_column(column_name,
                                                    shape=(input_size,))
    estimator = tf.estimator.DNNClassifier(
        hidden_units=network_params['hidden_units'],
        feature_columns=[point_column],
        # model_dir=network_params['model_dir']
        n_classes=output_size,
        activation_fn=network_params['activation'],
        optimizer=network_params['optimizer'])
    estimator.train(train_input_fn)

    # linear model for measuring degree of separability
    linear_model = tf.estimator.LinearClassifier(
        feature_columns=[point_column],
        n_classes=output_size)
    linear_model.train(total_input_train)
    linear_results = linear_model.evaluate(total_input_test)
    linear_results = {'linear_' + key: linear_results[key]
                      for key in linear_results}

    trial_results = dict(params)
    del trial_results['points']
    # TODO: record all info desired!
    trial_results['degree_of_convexity'] = the_partition.degree_of_convexity()
    trial_results.update(
        estimator.evaluate(test_input_fn))
    trial_results.update(
        {'cell{}_size'.format(label): len(part[label]) for label in part})
    trial_results.update(linear_results)
    print(trial_results)
    trial_results.update(
        {'cell_{}'.format(label): part[label] for label in part})

    return trial_results


def main_experiment(out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # set possible trial parameters
    temps = [5, 1, 0.1, 0.01, 0.001, 0.0005]
    convs = [0, 0.25, 0.5, 0.75, 1.0]
    num_labels = [7]
    num_epochs = [6]

    # how many trials per parameter combination to run
    trials_per_params = 10

    # get list of all parameters
    params_tuples = (list(
        itertools.product(temps, convs, num_labels, num_epochs)
    )*trials_per_params)

    tuple_labels = ['temp', 'conv', 'num_labels', 'num_epochs']

    # global parameters
    axis_stride = 0.05
    lab_points = partition.generate_CIELab_space(axis_stride=axis_stride)
    np.save(out_dir + 'points.npy', lab_points)

    trials_dicts = []

    # run the trials!
    for trial_tup in tqdm(params_tuples):
        params = dict(zip(tuple_labels, trial_tup))
        params['points'] = lab_points
        trials_dicts.append(
            run_trial(params, out_dir))
        # close files opened by that trial; otherwise, can wind up opening too
        # many files per experiment
        proc = psutil.Process()
        for handler in proc.open_files():
            os.close(handler.fd)

    trials_frame = pd.DataFrame(trials_dicts)
    trials_frame.to_csv(out_dir + 'results.csv')
    return trials_frame


if __name__ == '__main__':
    main_experiment('./trial/')
