---
title: Document Center
layout: doc
---

#Randomkit random number generators, wrapped for Torch

Provides and wraps the random nnumber generators the [Randomkit library](), copied from [Numpy]()

##Example

###Single sample

You can call any of the wrapped functions with just the distribution's parameters to generate a single sample and return a number:

```lua
require 'randomkit'
randomkit.poisson(5)
```

###Multiple samples from one distribution

Often, you might want to generate many samples identically distributed. Simply pass as a first argument a tensor of the proper dimension, into which the samples will be stored:

```lua
x = torch.Tensor(10000)
randomkit.poisson(x, 5)
```

The sampler returns the tensor, so you can shorten the above in:

```lua
x = randomkit.poisson(torch.Tensor(10000), 5)
```

###Multiple samples from multiple distributions

Finally, you might want to generate many samples, each from a distribution with different parameters. This is achieved by passing a Tensor as the parameter of the distribution:

```lua
many_lambda = torch.Tensor{5, 3, 40, 60}
x = randomkit.poisson(many_lambda)
```

Of course, this can be combined with passing a result Tensor as an optional first element, to re-use memory and avoid creaating a new Tensor at each call:

```lua
many_lambda = torch.Tensor{5, 3, 40, 60}
x = torch.Tensor(many_lambda:size())
randomkit.poisson(x, many_lambda)
```

**Note:** in the latter case, the size of the result Tensor must correspond to the size of the parameter tensor -- we do not resize the result tensor automatically, yet:


##Installation

From a terminal:

```bash
torch-rocks install randomkit
```

##List of Randomkit generators

See this **[extensive automatically extracted doc](randomkit.html)**, built from Numpy's docstrings.

##List of Torch-only generators

These functions are provided in addition to the scipy randomkit functions.

###Poisson

####randomkit.poissonPDF(x, lambda)

Probability density function of a Poisson distribution with mean `lambda`, evaluated at `x`.

####randomkit.poissonLogPDF(x, lambda)

Log of probability density function of a Poisson distribution with mean `lambda`, evaluated at `x`.

####randomkit.poissonCDF(x, lambda)

Cumulative distribution function of a Poisson distribution with mean `lambda`, evaluated at `x`.

###Gaussian

####randomkit.gaussianPDF(x, mu, sigma)

Probability density function of a Gaussian distribution with mean `mu` and standard deviation `sigma`, evaluated at `x`.

####randomkit.gaussianLogPDF(x, mu, sigma)

Log probability density function of a Gaussian distribution with mean `mu` and standard deviation `sigma`, evaluated at `x`.

####randomkit.gaussianCDF(x, mu, sigms)

Cumulative distribution function of a Gaussian distribution with mean `mu` and standard deviation `sigma`, evaluated at `x`.

###Multivariate Gaussian

The covariance matrix passed to multivariate gaussian functions **must** be definite positive: we do not deal with the degenerate case
where some of the variance elements are null.


####randomkit.multivariateGaussianPDF(x, mu, cov)

Probability density function of a multivariate Gaussian distribution with mean `mu` and covariance `cov`, evaluated at `x`.

For a D-dimensional Gaussian, the following forms are valid:

* `randomkit.multivariateGaussianPDF([D], [D], [D, D])` - returns a number.
* `randomkit.multivariateGaussianPDF([N, D], [D], [D, D])` - returns a Tensor.
* `randomkit.multivariateGaussianPDF([D], [N, D], [D, D])` - returns a Tensor.
* `randomkit.multivariateGaussianPDF([N, D], [N, D], [D, D])` - returns a Tensor.

In the case of a diagonal covariance `cov`, you may also opt to pass a vector containing only the diagonal elements:

* `randomkit.multivariateGaussianPDF([D], [D], [D])` - returns a number.
* `randomkit.multivariateGaussianPDF([N, D], [D], [D])` - returns a Tensor.
* `randomkit.multivariateGaussianPDF([D], [N, D], [D])` - returns a Tensor.
* `randomkit.multivariateGaussianPDF([N, D], [N, D], [D])` - returns a Tensor.

####randomkit.multivariateGaussianLogPDF(x, mu, cov)

Probability density function of a multivariate Gaussian distribution with mean `mu` and covariance `cov`, evaluated at `x`.


See `randomkit.multivariateGaussianPDF()` for description of valid forms for x, mu and cov.

####randomkit.multivariateGaussianRand([res,] mu, cov)

Sample from a multivariate Gaussian distribution with mean `mu` and covariance `cov`.

For a D-dimensional Gaussian, the following forms are valid:

* `randomkit.multivariateGaussianPDF([D], [D, D])` - returns 1 sample in a 1-by-D Tensor
* `randomkit.multivariateGaussianPDF([N, D], [D, D])` - returns N samples in a N-by-D Tensor
* `randomkit.multivariateGaussianPDF([N, D], [D], [D, D])` - stores and returns N samples in the N-by-D Tensor
* `randomkit.multivariateGaussianPDF([N, D], [N, D], [D, D])` - stores and returns N samples in the N-by-D Tensor

In the case of a diagonal covariance `cov`, you may also opt to pass a vector (not a matrix) containing only the diagonal elements.

###Cauchy

####randomkit.cauchyPDF(x, a, b)

Probability density function of a Cauchy distribution with location `a` and scale `b`, evaluated at `x`.

####randomkit.cauchyLogPDF(x, a, b)

Log of probability density function of a Cauchy distribution with location `a` and scale `b`, evaluated at `x`.

####randomkit.cauchyCDF(x, a, b)

Cumulative distribution function of a Cauchy distribution with location `a` and scale `b`, evaluated at `x`.

###Chi square

####randomkit.chi2PDF(x, dof)

Probability density function of a Chi square distribution with `dof` degrees of freedom, evaluated at `x`.

####randomkit.chi2LogPDF(x, dof)

Log of probability density function of a Chi square distribution with `dof` degrees of freedom, evaluated at `x`.

####randomkit.chi2CDF(x, dof)

Cumulative distribution function of a Chi square distribution with `dof` degrees of freedom, evaluated at `x`.

###Laplace

####randomkit.laplacePDF(x, loc, scale)

Probability density function of a Laplace distribution with location `loc` and scale `scale`, evaluated at `x`.

####randomkit.laplaceLogPDF(x, loc, scale)

Log of probability density function of a Laplace distribution with location `loc` and scale `scale`, evaluated at `x`.

####randomkit.laplaceCDF(x, loc, scale)

Cumulative distribution function of a Laplace distribution with location `loc` and scale `scale`, evaluated at `x`.

##Unit Tests

Last but not least, the unit tests are in the folder
[`luasrc/tests`](https://github.com/jucor/torch-randomkit/tree/master/luasrc/tests). You can run them from your local clone of the repostiory with:

```bash
git clone https://www.github.com/jucor/torch-randomkit
find torch-randomkit/luasrc/tests -name "test*lua" -exec torch {} \;
```

Those tests will soone be automatically installed with the package, once I sort out a bit of CMake resistance.

##Direct access to FFI

###randomkit.ffi.*

Functions directly accessible at the top of the `randomkit` table are Lua wrappers to the actual C functions from Randomkit, with extra error checking. If, for any reason, you want to get rid of this error checking and of a possible overhead, the FFI-wrapper functions can be called directly via `randomkit.ffi.myfunction()` instead of `randomkit.myfunction()`.

