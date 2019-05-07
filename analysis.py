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
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *


def analyze_results(filename, variables, fig_file=None):

    data = pd.read_csv(filename)
    # add a column of data for largest cell
    data['max_cell_size'] = data[[column for column in list(data) if 'size' in
        column]].max(axis=1)
    data['min_cell_size'] = data[[column for column in list(data) if 'size' in
        column]].min(axis=1)
    data['max_minus_min'] = data['max_cell_size'] - data['min_cell_size']
    data['min_over_max'] = data['min_cell_size'] / data['max_cell_size']
    data['median_size'] = data[[column for column in list(data) if 'size' in
        column]].median(axis=1)
    data['smooth*conn'] = data['smooth'] * data['conn']

    # correlations
    print(data.corr()['degree_of_convexity'])

    # ggplot version
    data['smooth'] = data['smooth'].astype('category')
    data['conn'] = data['conn'].astype('category')
    plot = (ggplot(data, aes(x='degree_of_convexity', y='accuracy'))
            + geom_point(aes(colour='conn', fill='smooth'), size=2.5)
            + geom_smooth(method='lm', colour='orange')
            + annotate('label', label='Pearson R: 0.711; p=1.9e-47',
                       x=0.2, y=0.9, size=14)
            + xlab('degree of convexity')
            + xlim((0, 1)) + ylim((0, 1)))
    if fig_file:
        plot.save(fig_file, width=18, height=12)
    else:
        print(plot)

    # regress all variables
    add_string = ' + '.join(variables)
    full_model = smf.ols(
        formula='accuracy ~ {}'.format(add_string),
        data=data)
    full_results = full_model.fit()
    full_r2 = full_results.rsquared
    print(full_results.summary())
    print(full_results.pvalues)

    r2_diffs = {}
    for variable in variables:
        # regress each variable
        single_model = smf.ols(
            formula='accuracy ~ {}'.format(variable),
            data=data)
        results = single_model.fit()
        print(results.summary())

        # regress every variable but this one
        all_but_this = list(variables)
        all_but_this.remove(variable)
        all_but_this_model = smf.ols(
            formula='accuracy ~ {}'.format(' + '.join(all_but_this)),
            data=data)
        all_but_this_results = all_but_this_model.fit()
        r2_diffs[variable] = full_r2 - all_but_this_results.rsquared

    print(r2_diffs)


if __name__ == '__main__':
    variables = ['degree_of_convexity', 'smooth', 'conn', 'max_cell_size',
                 'min_cell_size', 'max_minus_min', 'min_over_max',
                 'median_size', 'smooth*conn']
    analyze_results('data/results.csv', variables, fig_file='data/complex_regression.png')
    print('\n\n\nWITH LINEAR SEPARABILITY\n\n\n')
    analyze_results('data/results.csv', variables + ['linear_accuracy'], fig_file='data/complex_regression.png')
