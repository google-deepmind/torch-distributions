#torch-randomkit: extra functions

These functions are provided in addition to the scipy randomkit functions.

##Poisson

###randomkit.poissonPDF(x, lambda)

Probability density function of a Poisson distribution with mean `lambda`, evaluated at `x`.

###randomkit.poissonLogPDF(x, lambda)

Log of probability density function of a Poisson distribution with mean `lambda`, evaluated at `x`.

###randomkit.poissonCDF(x, lambda)

Cumulative distribution function of a Poisson distribution with mean `lambda`, evaluated at `x`.

##Gaussian

###randomkit.gaussianPDF(x, mu, sigma)

Probability density function of a Gaussian distribution with mean `mu` and standard deviation `sigma`, evaluated at `x`.

###randomkit.gaussianCDF(x, mu, sigma)

Cumulative distribution function of a Gaussian distribution with mean `mu` and standard deviation `sigma`, evaluated at `x`.

##Cauchy

###randomkit.cauchyPDF(x, a, b)

Probability density function of a Cauchy distribution with location `a` and scale `b`, evaluated at `x`.

###randomkit.cauchyLogPDF(x, a, b)

Log of probability density function of a Cauchy distribution with location `a` and scale `b`, evaluated at `x`.

###randomkit.cauchyCDF(x, a, b)

Cumulative distribution function of a Cauchy distribution with location `a` and scale `b`, evaluated at `x`.

##Chi square

###randomkit.chi2PDF(x, dof)

Probability density function of a Chi square distribution with `dof` degrees of freedom, evaluated at `x`.

###randomkit.chi2LogPDF(x, dof)

Log of probability density function of a Chi square distribution with `dof` degrees of freedom, evaluated at `x`.

###randomkit.chi2CDF(x, dof)

Cumulative distribution function of a Chi square distribution with `dof` degrees of freedom, evaluated at `x`.

##Laplace

###randomkit.laplacePDF(x, loc, scale)

Probability density function of a Laplace distribution with location `loc` and scale `scale`, evaluated at `x`.

###randomkit.laplaceLogPDF(x, loc, scale)

Log of probability density function of a Laplace distribution with location `loc` and scale `scale`, evaluated at `x`.

###randomkit.laplaceCDF(x, loc, scale)

Cumulative distribution function of a Laplace distribution with location `loc` and scale `scale`, evaluated at `x`.
