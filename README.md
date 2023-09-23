# Bayesian Linear Regression Implementation in MATLAB

This repository contains a MATLAB implementation of Bayesian Linear Regression based on the concepts presented in the book "Pattern Recognition and Machine Learning" by Christopher M. Bishop.

## Overview

Bayesian Linear Regression is a probabilistic approach to linear regression that provides not only point estimates of the model parameters but also their uncertainties. 
This implementation follows the methods and techniques described in Bishop's book and provides two main MATLAB files for performing Bayesian Linear Regression.

## Files

1. **Bayesian_Linear_Regression.m**: Contains the main implementation of Bayesian Linear Regression.
2. **parameter_optimization.m**: A script for learning the hyperparameters $\beta$ and $\alpha$ from the data.
3. **data.txt**:  A provided dataset to test and evaluate the implementation.

## Getting Started

1. Clone this repository.
  
2. Open MATLAB and navigate to the cloned directory.

3. Run `Bayesian_Linear_Regression.m` to perform Bayesian Linear Regression on your data.

## Dataset

In this repository, you'll find a dataset file named "data.txt" containing N=30 noisy samples of the function $f(x) = cos(2 \pi \chi) - (3\chi - 2)^2$.
The dataset was generated within the range [0, 1] with added noise.

### File Format

The "data.txt" file is formatted as follows:

- Each row represents a data point, consisting of two columns: $x_i$ and $t_i$.
- The first column, $x_i$, represents the input value.
- The second column, $t_i$, represents the corresponding noisy target value.
  
### Acknowledgments
 - Christopher M. Bishop for his book "Pattern Recognition and Machine Learning," which serves as the foundation for this implementation.


