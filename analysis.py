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


def analyze_results(filename):

    data = pd.read_csv(filename)
    # add a column of data for largest cell
    data['max_cell_size'] = data[[column for column in list(data) if 'size' in
        column]].max(axis=1)
    data['min_cell_size'] = data[[column for column in list(data) if 'size' in
        column]].min(axis=1)
    data['max_minus_min'] = data['max_cell_size'] - data['min_cell_size']
    data['max_over_min'] = data['max_cell_size'] / data['min_cell_size']
    data['median_size'] = data[[column for column in list(data) if 'size' in
        column]].median(axis=1)
    data['temp:conv'] = data['temp'] * data['conv']

    # correlations
    print(data.corr())

    # joint plot with regression
    sns.set_palette('colorblind')
    sns.jointplot(x='degree_of_convexity', y='accuracy', data=data, kind='reg')
    plt.show()

    # variables of interest
    variables = ['degree_of_convexity', 'temp', 'conv', 'max_cell_size',
                 'min_cell_size', 'max_minus_min', 'max_over_min',
                 'median_size', 'temp:conv']

    # regress all variables
    add_string = ' + '.join(variables)
    full_model = smf.ols(
        formula='accuracy ~ {}'.format(add_string),
        data=data)
    full_results = full_model.fit()
    full_r2 = full_results.rsquared
    print(full_results.summary())

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

    """
    # linear regression
    stat_model = smf.ols(
            formula=('accuracy ~ degree_of_convexity'),
            data=data)
    results = stat_model.fit()
    print(results.summary())

    stat_model = smf.ols(
            formula=('accuracy ~ degree_of_convexity + max_cell_size'),
            data=data)
    results = stat_model.fit()
    print(results.summary())
    print(results.f_test('max_cell_size = 0'))

    stat_model = smf.ols(
        formula=('accuracy ~ degree_of_convexity + degree_of_convexity:max_cell_size'),
            data=data)
    results = stat_model.fit()
    print(results.summary())
    print(results.f_test('degree_of_convexity:max_cell_size = 0'))

    stat_model = smf.ols(
            formula=('accuracy ~ degree_of_convexity + max_cell_size +'
                'degree_of_convexity:max_cell_size'),
            data=data)
    results = stat_model.fit()
    print(results.summary())
    print(results.pvalues)
    print(results.f_test('max_cell_size = 0, degree_of_convexity:max_cell_size=0'))
    print(results.f_test('max_cell_size = 0'))

    plt.plot(data.degree_of_convexity, data.accuracy, 'o')
    plt.show()
    sm.graphics.plot_fit(results, 'degree_of_convexity')
    plt.show()
    sm.graphics.plot_regress_exog(results, 'degree_of_convexity')
    plt.show()
    """

    # TODO: other models, plots
    """
    sm.graphics.plot_partregress('accuracy', 'degree_of_convexity',
            ['max_cell_size', 'degree_of_convexity:max_cell_size'],
            data=data)
    plt.show()
    """


if __name__ == '__main__':
    analyze_results('results.csv')
