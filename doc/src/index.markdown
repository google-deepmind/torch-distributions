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

###Poisson: poisson

####poisson.pdf(x, lambda)

Probability density function of a Poisson distribution with mean `lambda`, evaluated at `x`.

####poisson.logpdf(x, lambda)

Log of probability density function of a Poisson distribution with mean `lambda`, evaluated at `x`.

####poisson.cdf(x, lambda)

Cumulative distribution function of a Poisson distribution with mean `lambda`, evaluated at `x`.

###Normal/Gaussian: norm

####norm.pdf(x, mu, sigma)

Probability density function of a Normal distribution with mean `mu` and standard deviation `sigma`, evaluated at `x`.

####norm.logpdf(x, mu, sigma)

Log probability density function of a Normal distribution with mean `mu` and standard deviation `sigma`, evaluated at `x`.

####norm.cdf(x, mu, sigma)

Cumulative distribution function of a Normal distribution with mean `mu` and standard deviation `sigma`, evaluated at `x`.

###Multivariate Normal: mvn

The covariance matrix passed to multivariate gaussian functions needs only be positive **semi**-definite: we deal gracefully with the degenerate case of rank-deficient covariance. 

Those functions also accept the upper-triangular Cholesky decomposition instead, by setting the field `cholesky = true` in the optional table `options`.

####mnv.pdf(x, mu, M, [options])

Probability density function of a multivariate Normal distribution with mean `mu` and covariance or cholesky of the covariance specified in `M`, evaluated at `x`. 

By defaut, the matrix `M` is the covariance matrix. However, it is possible to pass the upper-triangular Cholesky decomposition instead, by setting the field `cholesky = true` in the optional table `options`.

For a D-dimensional Normal, the following forms are valid:

* `mvn.pdf([D], [D], [D, D])` - returns a number.
* `mvn.pdf([N, D], [D], [D, D])` - returns a Tensor.
* `mvn.pdf([D], [N, D], [D, D])` - returns a Tensor.
* `mvn.pdf([N, D], [N, D], [D, D])` - returns a Tensor.

In the case of a diagonal covariance `cov`, you may also opt to pass a vector containing only the diagonal elements:

* `mvn.pdf([D], [D], [D])` - returns a number.
* `mvn.pdf([N, D], [D], [D])` - returns a Tensor.
* `mvn.pdf([D], [N, D], [D])` - returns a Tensor.
* `mvn.pdf([N, D], [N, D], [D])` - returns a Tensor.

####mvn.logpdf(x, mu, M, [options])

Probability density function of a multivariate Normal distribution with mean `mu` and covariance matrix `M`, evaluated at `x`.


See `mvn.pdf()` for description of valid forms for x, mu and cov and options.

####mvn.rnd([res,] mu, M, [options])

Sample from a multivariate Normal distribution with mean `mu` and covariance matrix `M`.

For a D-dimensional Normal, the following forms are valid:

* `mvn.rnd([D], [D, D])` - returns 1 sample in a 1-by-D Tensor
* `mvn.rnd([N, D], [D, D])` - returns N samples in a N-by-D Tensor
* `mvn.rnd([N, D], [D], [D, D])` - stores and returns N samples in the N-by-D Tensor
* `mvn.rnd([N, D], [N, D], [D, D])` - stores and returns N samples in the N-by-D Tensor

In the case of a diagonal covariance `cov`, you may also opt to pass a vector (not a matrix) containing only the diagonal elements.

By defaut, the matrix `M` is the covariance matrix. However, it is possible to pass the upper-triangular Cholesky decomposition instead, by setting the field `cholesky = true` in the optional table `options`.

###Categorical: cat

Categorical distributions on indices from 1 to K = p:numel()

Not vectorized in p.

####cat.pdf(x, p, [options])

Not implemented

####mvn.logpdf(x, p, [options])

Not implemented

####mvn.rnd([res|N,] p, [options])

Sample `N = size(res,1)` amongst `K = 1 ... p:numel()`, where the probability of category k is given by p[k]/p:sum().

Options is a table containing:

* options.type Type of sampler:
    - `nil` or `'iid'`: default, i.i.d samples, use linear search in O(N log N + max(K, N)), best when K/N is close to 1.
    - 'dichotomy': dichotomic search, same variance, faster when small K large N
    - 'stratified': sorted stratified samples, sample has lower variance than i.i.d. but not independent, best when K/N is close to 1

Returns a LongTensor vector with N elements, or store into the given result tensor.

###Cauchy: cauchy

####cauch.pdf(x, a, b)

Probability density function of a Cauchy distribution with location `a` and scale `b`, evaluated at `x`.

####cauchy.logpdf(x, a, b)

Log of probability density function of a Cauchy distribution with location `a` and scale `b`, evaluated at `x`.

####cauchy.cdf(x, a, b)

Cumulative distribution function of a Cauchy distribution with location `a` and scale `b`, evaluated at `x`.

###Chi square: chi2

####chi2.pdf(x, dof)

Probability density function of a Chi square distribution with `dof` degrees of freedom, evaluated at `x`.

####chi2.logpdf(x, dof)

Log of probability density function of a Chi square distribution with `dof` degrees of freedom, evaluated at `x`.

####chi2.cdf(x, dof)

Cumulative distribution function of a Chi square distribution with `dof` degrees of freedom, evaluated at `x`.

###Laplace: laplace

####laplage.pdf(x, loc, scale)

Probability density function of a Laplace distribution with location `loc` and scale `scale`, evaluated at `x`.

####laplace.logpdf(x, loc, scale)

Log of probability density function of a Laplace distribution with location `loc` and scale `scale`, evaluated at `x`.

####laplace.cdf(x, loc, scale)

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

Perform a chi-squared test, with null hypothesis "sample x is from a Normal distribution with mean `mu` and variance `sigma`".

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
