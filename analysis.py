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
import matplotlib.pyplot as plt

def analyze_results(filename):

    data = pd.read_csv(filename)
    # add a column of data for largest cell
    data['max_cell_size'] = data[[column for column in list(data) if 'size' in
        column]].max(axis=1)

    # linear regression
    stat_model = smf.ols(
            formula=('accuracy ~ degree_of_convexity + max_cell_size +'
                'degree_of_convexity:max_cell_size'),
            data=data)
    results = stat_model.fit()
    print(results.summary())

    sm.graphics.plot_fit(results, 'degree_of_convexity')
    plt.show()
    sm.graphics.plot_regress_exog(results, 'degree_of_convexity')
    plt.show()

    # TODO: other models, plots
    """
    sm.graphics.plot_partregress('accuracy', 'degree_of_convexity',
            ['max_cell_size', 'degree_of_convexity:max_cell_size'],
            data=data)
    plt.show()
    """


if __name__ == '__main__':
    analyze_results('results.csv')
