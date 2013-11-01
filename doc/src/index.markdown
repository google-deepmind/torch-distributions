---
title: Document Center
layout: doc
---

#Randomkit random number generators, wrapped for Torch

* Toc will go here
{:toc}

Provides and wraps the random nnumber generators the [Randomkit library](), copied from [Numpy]()

##Installation

From a terminal:
{% highlight bash %}
torch-rocks install https://raw.github.com/jucor/torch-randomkit/master/randomkit-0-0.rockspec
{% endhighlight %}

This will be simplified once the rockspec will be merged into the list of official rocks.

##List of Randomkit generators

See this [automatically extracted doc](randomkit.html), built from Numpy's docstrings.

##Added functions

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

####randomkit.multivariateGaussianPDF(x, mu, sigma)

Probability density function of a multivariate Gaussian distribution with mean `mu` and covariance `sigma`, evaluated at `x`.

For a Gaussian with D variables, the following forms are valid:

`randomkit.multivariateGaussianPDF([D], [D], [D, D])` - returns a number.
`randomkit.multivariateGaussianPDF([N, D], [D], [D, D])` - returns a Tensor.
`randomkit.multivariateGaussianPDF([D], [N, D], [D, D])` - returns a Tensor.
`randomkit.multivariateGaussianPDF([N, D], [N, D], [D, D])` - returns a Tensor.

In the case of a diagonal covariance `sigma`, you may also opt to pass a vector containing only the diagonal elements:

`randomkit.multivariateGaussianPDF([D], [D], [D])` - returns a number.
`randomkit.multivariateGaussianPDF([N, D], [D], [D])` - returns a Tensor.
`randomkit.multivariateGaussianPDF([D], [N, D], [D])` - returns a Tensor.
`randomkit.multivariateGaussianPDF([N, D], [N, D], [D])` - returns a Tensor.

####randomkit.multivariateGaussianLogPDF(x, mu, sigma)

Probability density function of a multivariate Gaussian distribution with mean `mu` and covariance `sigma`, evaluated at `x`.

See `randomkit.multivariateGaussianPDF()` for description of valid forms for x, mu and sigma.

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
(`luasrc/tests`)[https://github.com/jucor/torch-randomkit/tree/master/luasrc/tests]. You can run them from your local clone of the repostiory with:

{% highlight bash %}
git clone https://www.github.com/jucor/torch-randomkit
find torch-randomkit/luasrc/tests -name "test&ast;lua" -exec torch {} \;
{% endhighlight %}

Those tests will soone be automatically installed with the package, once I sort out a bit of CMake resistance.

##Direct access to FFI

###randomkit.ffi.&ast;

Functions directly accessible at the top of the `randomkit` table are Lua wrappers to the actual C functions from Randomkit, with extra error checking. If, for any reason, you want to get rid of this error checking and of a possible overhead, the FFI-wrapper functions can be called directly via `randomkit.ffi.myfunction()` instead of `randomkit.myfunction()`.
