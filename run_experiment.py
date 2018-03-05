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

import numpy as np
import tensorflow as tf
import pandas as pd

import partition


def run_trial(params, out_dir):

    # model input and output sizes
    input_size = params['points'].shape[1]
    output_size = params['num_labels']
    column_name = 'point'

    # TODO: prep input: (i) split bins into train/test, (ii) oversample
    # training bins, (iii) make into feature column
    # note: partition.labelled_pts will no longer be true labels for training
    # data, only for test
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
    max_train_bin = max(len(train_bins[label]) for label in train_bins)
    train_bins = {label: np.random.choice(train_bins[label], max_train_bin,
                                          replace=True)
                  if len(train_bins[label]) < max_train_bin else
                  train_bins[label] for label in train_bins}

    def from_bins_to_xy(bins):
        x = np.vstack([points[bins[label]] for label in bins])
        y = np.concatenate([[label]*len(bins[label]) for label in bins])
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

    # TODO: vary network by trial or not?
    network_params = {
        'hidden_units': [12, 12],
        'activation': tf.nn.elu,
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

    trial_results = dict(params)
    del trial_results['points']
    # TODO: record all info desired!
    trial_results['degree_of_convexity'] = the_partition.degree_of_convexity()
    trial_results.update(
        estimator.evaluate(test_input_fn))
    print(trial_results)
    return trial_results


def main_experiment(out_dir):

    # set possible trial parameters
    temps = [5, 1, 0.1, 0.01, 0.001, 0.0005]
    convs = [0, 0.25, 0.5, 0.75, 1.0]
    temps = [0.001]
    convs = [0.75, 1.0]
    num_labels = [7]
    num_epochs = [5]

    # how many trials per parameter combination to run
    trials_per_params = 1

    # get list of all parameters
    params_tuples = (list(
        itertools.product(temps, convs, num_labels, num_epochs)
    )*trials_per_params)

    tuple_labels = ['temp', 'conv', 'num_labels', 'num_epochs']

    # global parameters
    axis_stride = 0.075
    lab_points = partition.generate_CIELab_space(axis_stride=axis_stride)

    trials_dicts = []

    # run the trials!
    for trial_tup in params_tuples:
        params = dict(zip(tuple_labels, trial_tup))
        params['points'] = lab_points
        trials_dicts.append(
            run_trial(params, out_dir))

    trials_frame = pd.DataFrame(trials_dicts)
    trials_frame.to_csv(out_dir + 'results.csv')
    return trials_frame


if __name__ == '__main__':
    main_experiment('data/')
