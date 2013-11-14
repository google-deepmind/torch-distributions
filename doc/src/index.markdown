---
title: Probability distributions for Torch
layout: doc
---

#Probability distributions for Torch

##Example

TODO

###Getting/setting the seed and the state

Distributions is transparently integrated with Torch's random stream: just use `torch.manualSeed(seed)`, `torch.getRNGState()`, and `torch.setRNGState(state)` as usual.


##Installation

From a terminal:

```bash
torch-rocks install distributions
```

##List of Distributions

###Poisson

####poissonPDF(x, lambda)

Probability density function of a Poisson distribution with mean `lambda`, evaluated at `x`.

####poissonLogPDF(x, lambda)

Log of probability density function of a Poisson distribution with mean `lambda`, evaluated at `x`.

####poissonCDF(x, lambda)

Cumulative distribution function of a Poisson distribution with mean `lambda`, evaluated at `x`.

###Gaussian

####gaussianPDF(x, mu, sigma)

Probability density function of a Gaussian distribution with mean `mu` and standard deviation `sigma`, evaluated at `x`.

####gaussianLogPDF(x, mu, sigma)

Log probability density function of a Gaussian distribution with mean `mu` and standard deviation `sigma`, evaluated at `x`.

####gaussianCDF(x, mu, sigms)

Cumulative distribution function of a Gaussian distribution with mean `mu` and standard deviation `sigma`, evaluated at `x`.

###Multivariate Gaussian

The covariance matrix passed to multivariate gaussian functions needs only be positive **semi**-definite: we deal gracefully with the degenerate case of rank-deficient covariance.

####multivariateGaussianPDF(x, mu, cov)

Probability density function of a multivariate Gaussian distribution with mean `mu` and covariance `cov`, evaluated at `x`.

For a D-dimensional Gaussian, the following forms are valid:

* `multivariateGaussianPDF([D], [D], [D, D])` - returns a number.
* `multivariateGaussianPDF([N, D], [D], [D, D])` - returns a Tensor.
* `multivariateGaussianPDF([D], [N, D], [D, D])` - returns a Tensor.
* `multivariateGaussianPDF([N, D], [N, D], [D, D])` - returns a Tensor.

In the case of a diagonal covariance `cov`, you may also opt to pass a vector containing only the diagonal elements:

* `multivariateGaussianPDF([D], [D], [D])` - returns a number.
* `multivariateGaussianPDF([N, D], [D], [D])` - returns a Tensor.
* `multivariateGaussianPDF([D], [N, D], [D])` - returns a Tensor.
* `multivariateGaussianPDF([N, D], [N, D], [D])` - returns a Tensor.

####multivariateGaussianLogPDF(x, mu, cov)

Probability density function of a multivariate Gaussian distribution with mean `mu` and covariance `cov`, evaluated at `x`.


See `multivariateGaussianPDF()` for description of valid forms for x, mu and cov.

####multivariateGaussianRand([res,] mu, cov)

Sample from a multivariate Gaussian distribution with mean `mu` and covariance `cov`.

For a D-dimensional Gaussian, the following forms are valid:

* `multivariateGaussianPDF([D], [D, D])` - returns 1 sample in a 1-by-D Tensor
* `multivariateGaussianPDF([N, D], [D, D])` - returns N samples in a N-by-D Tensor
* `multivariateGaussianPDF([N, D], [D], [D, D])` - stores and returns N samples in the N-by-D Tensor
* `multivariateGaussianPDF([N, D], [N, D], [D, D])` - stores and returns N samples in the N-by-D Tensor

In the case of a diagonal covariance `cov`, you may also opt to pass a vector (not a matrix) containing only the diagonal elements.

###Cauchy

####cauchyPDF(x, a, b)

Probability density function of a Cauchy distribution with location `a` and scale `b`, evaluated at `x`.

####cauchyLogPDF(x, a, b)

Log of probability density function of a Cauchy distribution with location `a` and scale `b`, evaluated at `x`.

####cauchyCDF(x, a, b)

Cumulative distribution function of a Cauchy distribution with location `a` and scale `b`, evaluated at `x`.

###Chi square

####chi2PDF(x, dof)

Probability density function of a Chi square distribution with `dof` degrees of freedom, evaluated at `x`.

####chi2LogPDF(x, dof)

Log of probability density function of a Chi square distribution with `dof` degrees of freedom, evaluated at `x`.

####chi2CDF(x, dof)

Cumulative distribution function of a Chi square distribution with `dof` degrees of freedom, evaluated at `x`.

###Laplace

####laplacePDF(x, loc, scale)

Probability density function of a Laplace distribution with location `loc` and scale `scale`, evaluated at `x`.

####laplaceLogPDF(x, loc, scale)

Log of probability density function of a Laplace distribution with location `loc` and scale `scale`, evaluated at `x`.

####laplaceCDF(x, loc, scale)

Cumulative distribution function of a Laplace distribution with location `loc` and scale `scale`, evaluated at `x`.

##Hypothesis Testing

Besides the generators, there are some functions for checking whether a sample fits a particular distribution, using [`Pearson's chi-squared test`](http://en.wikipedia.org/wiki/Pearson's_chi-squared_test).

###chi2Uniform(x, [low, up, nBins])

Perform a chi-squared test, with null hypothesis "sample x is from a continuous uniform distribution on the interval `[low, up]`".

* `x` should be a vector of sample values to test
* `low` is the lower end of the uniform distribution's support interval (default: 0)
* `up` is the upper end of the uniform distribution's support interval (default: 1)
* `nBins` is number of frequency buckets to use for the test (default: 100)

Returns: `p`, `chi2` - the p-value and the chi-squared score of the test, respectively.

###chi2TestCDF(x, cdf, cdfParams, [nBins])

Perform a chi-squared test, with null hypothesis "sample x is from a distribution with cdf `cdf`, parameterised by `cdfParams`".

* `x` should be a vector of sample values to test
* `cdf` should be a function which takes a number of parameters followed by a sample value and returns the cumulative density of the distribution up to that point
* `cdfParams` should be a table of parameters which will be passed to `cdf`
* `nBins` is number of frequency buckets to use for the test (default: 100)

Returns: `p`, `chi2` - the p-value and the chi-squared score of the test, respectively.

###chi2Gaussian(x, mu, sigma, [nBins])

Perform a chi-squared test, with null hypothesis "sample x is from a Gaussian distribution with mean `mu` and variance `sigma`".

* `x` should be a vector of sample values to test
* `mu` should be a number - the mean
* `sigma` should be a positive number - the variance
* `nBins` is number of frequency buckets to use for the test (default: 100)

Returns: `p`, `chi2` - the p-value and the chi-squared score of the test, respectively.

##Unit Tests

Last but not least, the unit tests are in the folder
[`luasrc/tests`](https://github.com/jucor/torch-distributions/tree/master/luasrc/tests). You can run them from your local clone of the repostiory with:

```bash
git clone https://www.github.com/jucor/torch-distributions
find torch-distributions/luasrc/tests -name "test*lua" -exec torch {} \;
```

Those tests will soone be automatically installed with the package, once I sort out a bit of CMake resistance.
