require 'cephes'

--[[ Chi-square test for uniformity, with known lower and upper bound.

One sample Chi-square test with uniform distribution with the specified support.

Arguments:

* `x` (1D Tensor) tensor of samples
* `low` (default 0) lower end of support interval
* `up` (default 1) upper end of support interval
* `nBins` (default 100) number of bins

Returns:

1. p-value of the test
2. value of the test statistic
]]
function distributions.chi2Uniform(x, low, up, nBins)
    if not distributions._isTensor(x) then
        error("chi2Uniform: expected tensor of samples as first argument")
    end
    if not x:nDimension() == 1 then
        error("chi2Uniform: tensor of samples should have only one dimension")
    end
    low = low or 0
    up = up or 1
    nBins = nBins or 100

    -- 10 bins per integer between low (included) and up (excluded)
    local actualBins = {}
    for i=1,nBins  do
        actualBins[i] = 0
    end
    local nPoints = x:size(1)
    for k = 1,nPoints do
        local v = x[k]
        local idx = 1+math.floor(nBins*(v-low)/(up-low))
        if idx < 1 or idx > nBins  or v < low or v > up then
            -- If out of support of the uniform, p-value is 0
            return 0, math.huge
        end
        actualBins[idx] = actualBins[idx] + 1
    end

    local expected = nPoints / nBins
    local chi2 = 0
    for i=1,nBins  do
        chi2 = chi2 + (expected - actualBins[i])^2 / expected
    end

    local df = nBins - 1
    return 1 - cephes.ffi.chdtr(df, chi2), chi2
end

--[[ Chi square test for arbitrary cdf

One sample Chi-square test with known CDF.

Arguments:

* `x` (1D Tensor) tensor of samples
* `cdf` (function) cumulative density function under the null hypothesis which
    takes sample tensor as its last argument
* `cdfParams` (table) table of parameters to the cdf function minus the samples
* `nBins` (default 100) number of bins

Returns:

1. p-value of the test
2. value of the test statistic
]]
function distributions.chi2TestCDF(x, cdf, cdfParams, nBins)
    if not distributions._isTensor(x) then
        error("chi2TestCDF requires a tensor of samples as its first argument")
    end
    if not (type(cdf) == 'function') then
        error("chi2TestCDF requires a cumulative distribution function as" ..
               " its second argument")
    end
    if not (type(cdfParams) == 'table') then
        error("chi2TestCDF requires a table of distribution parameters as" ..
              " its third argument")
    end
    cdfParams[#cdfParams+1] = x
    local transformed = cdf(unpack(cdfParams))
    return distributions.chi2Uniform(transformed, 0, 1, nBins)
end

--[[ Chi square test for gaussian distribution

One sample Chi-square test with gaussian.

Arguments:

* `x` (1D Tensor) tensor of samples
* `mu` (number) mean under the null hypothesis
* `sigma` (number > 0) standard deviation under the null hypothesis
* `nBins` (default 100) number of bins

Returns:

1. p-value of the test
2. value of the test statistic
]]
function distributions.chi2Gaussian(x, mu, sigma, nBins)
    local function gaussianCDF(x)
        return cephes.ndtr((x-mu) / sigma)
    end

    return distributions.chi2TestCDF(x, gaussianCDF, {}, nBins)
end

