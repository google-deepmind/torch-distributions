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
luarocks install https://raw.github.com/jucor/torch-distributions/master/distributions-0-0.rockspec
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

####mvn.pdf(x, mu, M, [options])

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

###Categorical/Multinomial: cat

Categorical distributions on indices from 1 to K = p:numel().

Not vectorized in p. See mvcat for vectorized version.

####cat.pdf(x, p, [options])

Not implemented

####cat.logpdf(x, p, [options])

Not implemented

####cat.rnd([res|N,] p, [options])

Sample `N = size(res,1)` amongst `K = 1 ... p:numel()`, where the probability of category k is given by p[k]/p:sum().

Options is a table containing:

* options.type Type of sampler:
    - `nil` or `'iid'`: default, i.i.d samples, use linear search in O(N log N + max(K, N)), best when K/N is close to 1.
    - 'dichotomy': dichotomic search, same variance, faster when small K large N
    - 'stratified': sorted stratified samples, sample has lower variance than i.i.d. but not independent, best when K/N is close to 1

* options.categories Categories to sample from
    - `nil`: default, returns integers between 1 and K
    - K-by-D tensor: each row is a category, must have has many rows as p:numel()

Returns a LongTensor vector with N elements in the resulting tensor if no categories is given,
or a new tensor of N rows corresponding to the categories given.

Note that it is not yet possible to use a result tensor *and* categories at the same time. This will be possible once [torch's index() accepts result tensor](https://github.com/torch/torch7-distro/issues/202).

###Multiple Categorical: mvcat

Vectorized version of `cat`, where `p` is now a matrix where each row represents a vector of probabilities. It samples independently for each row of `p`.

####mvcat.pdf(x, p, [options])

Not implemented

####mvcat.logpdf(x, p, [options])

Not implemented

####mvcat.rnd([res|N,] p, [options])

For each row `r = 1 ... R` of the matrix `p`, sample `N = size(res, 2)` amongst `K = 1 ... p:size(2)`, where the probability of category k is given by p[r][k]/p:sum(1).

Options is a table containing:

* options.type Type of sampler:
    - `nil` or `'iid'`: default, i.i.d samples, use linear search in O(N log N + max(K, N)), best when K/N is close to 1.
    - 'dichotomy': dichotomic search, same variance, faster when small K large N
    - 'stratified': sorted stratified samples, sample has lower variance than i.i.d. but not independent, best when K/N is close to 1


Returns a LongTensor vector with R-by-N elements in the resulting tensor.
or a new tensor of R rows with N columns corresponding to the categories given.

Note that `mvcat`, unlike `cat`, only returns tensor of integers: it does not allow for specifying a tensor of categories, to keep the handling of dimensions simple.

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

Besides the generators, there are some functions for checking whether two samples come from the same unspecified distribution using [`Kolmogorov-Smirnov two-sample test`](http://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov.E2.80.93Smirnov_test), and whether a sample fits a particular distribution, using [`Pearson's chi-squared test`](http://en.wikipedia.org/wiki/Pearson's_chi-squared_test).

###kstwo(x1, x2)

Perform a two-sample Kolmogorov-Smirnov test, with null hypothesis "sample x1 and sample x2 come from the same distribution".

* `x1` should be a vector of sample values to test 
* `x2` should be a vector of sample values to test 

Returns: `p`, `d` - the p-value and the statistic the test, respectively.

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

