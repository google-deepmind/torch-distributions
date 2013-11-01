require 'cephes'

local function isTensor(v)
    if torch.typename(v) then
        return string.sub(torch.typename(v), -6, -1) == "Tensor"
    end
end

--[[! Chi-square test for uniformity, with known lower and upper bound

One sample Chi-square test with uniform distribution with the specified support.

@param x table of samples
@param low lower end of support interval
@param up upper end of support interval
@param nBins number of bins

@return p-value of the test
@return value of the test statistic
]]
function randomkit.chi2Uniform(x, low, up, nBins)
    if not isTensor(x) then
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
    local nPoints = 0
    for k = 1,x:size(1) do
        local v = x[k]
        local idx = 1+math.floor(nBins*(v-low)/(up-low))
        if idx < 1 or idx > nBins  or v < low or v > up then
            -- If out of support of the uniform, p-value is 0
            return 0
        end
        actualBins[idx] = actualBins[idx] + 1
        nPoints = nPoints + 1
    end

    local expected = nPoints / nBins
    local chi2 = 0
    for i=1,nBins  do
        chi2 = chi2 + (expected - actualBins[i])^2 / expected
    end

    local df = nBins - 1
    return 1 - cephes.ffi.chdtr(df, chi2), chi2
end

--[[! Chi square test for arbitrary cdf

One sample Chi-square test with known CDF.

@param x table of samples
@param cdf cumulative density function under the null hypothesis
@param ... see chi2Uniform()

@return same as chi2Uniform()
--]]
function randomkit.chi2TestCDF(x, cdf, cdfParams, ...)
    if not isTensor(x) then
        error("chi2TestCDF requires a tensor of samples as its first argument")
    end
    if not (type(cdf) == 'function') then
        error("chi2TestCDF requires a cumulative distribution function as its second argument")
    end
    if not (type(cdfParams) == 'table') then
        error("chi2TestCDF requires a table of distribution parameters as its third argument")
    end
    cdfParams[#cdfParams+1] = x
    local transformed = cdf(unpack(cdfParams))
    return randomkit.chi2Uniform(transformed, 0, 1, ...)
end

--[[! Chi square test for gaussian distribution

One sample Chi-square test with gaussian.

@param x table of samples
@param mu mean under the null hypothesis
@param sigma standard deviation under the null hypothesis
@param ... see chi2Uniform()

@return same as chi2Uniform()
--]]
function randomkit.chi2Gaussian(x, mu, sigma, ...)
    local function gaussianCDF(x)
        return cephes.ndtr((x-mu) / sigma)
    end

    return randomkit.chi2TestCDF(x, gaussianCDF, {}, ...)
end

