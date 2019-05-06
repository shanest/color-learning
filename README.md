# Ease of Learning Explains Semantic Universals

This repository accompanies the paper:
* Shane Steinert-Threlkeld and Jakub Szymanik, "Ease of Learning Explains Semantic Universals", under revision at _Cognition_. https://semanticsarchive.net/Archive/zM5ZGIxM/EaseLearning.pdf

We generate artificial color naming systems by partitioning the CIELab color space, then train neural networks to learn those systems.  We show that the ability of networks to learn these color systems is well-explained by their _degree of convexity_, supporting a semantic universal for color terms across language.

## Requirements

Python 2.7, TensorFlow 1.10, Numpy 1.13, Pandas

(NB:  the code should be compatible with Python 3, but has not been tested with it. Later versions of TF and NP will also likely work, but no promises.)

## Running Experiments

Calling
```
python run_experiment.py
```
will run the main experiment reported in the paper.  In the output directory `trial`, the following will be output:
* `results.csv`: each row is one trial, recoring the color system generating algorithm's parameter values, degree of convexity, network accuracy, as well as other geometric variables, and the actual partition
* `points.npy`: a numpy array, with the CIELab space points

To see how an experiment is defined and to play with defining your own, you can check out the [`main_experiment`](https://github.com/shanest/color-learning/blob/master/run_experiment.py#L141) method, which itself repeatedly calls [`run_trial`](https://github.com/shanest/color-learning/blob/master/run_experiment.py#L31).

The algorithm for generating artificial color naming systems (and some utility methods for it) are in `partition.py`.

## Analyzing the Data

The regression and commonality analysis reported can be run via

```
python analysis.py
```
This will also produce `trial/complex_regression.png`.

The cluster analysis is handled by `clusters.R`, an will produce the cluster PNG files.
