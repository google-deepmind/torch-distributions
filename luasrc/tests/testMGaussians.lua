require 'randomkit'
require 'util.warn'
local N = 10000
local M = 3
local D = 2

local myTests = {}
local tester = torch.Tester()
torch.manualSeed(os.clock())

-- This is the confidence threshold for the statistical tests.
local function bonferroniCorrection(alpha, n)
    return 1 - (1 - alpha)/n
end
local statisticalTests = 28
local alpha = bonferroniCorrection(0.5, statisticalTests)

local testCount = 0

local function statisticalTestMultivariateGaussian(samples, mu, sigma, shouldAccept)

    testCount = testCount + 1

    -- Part one: chi2 test projection onto each axis

    assert(samples:dim() == 2)
    local N = samples:size(1)
    local D = samples:size(2)

    assert(mu:dim() == 1)
    assert(mu:size(1) == D)

    assert(sigma:dim() == 2)
    assert(sigma:size(1) == D)
    assert(sigma:size(2) == D)

    local rejectionCount = 0
    for k = 1, D do
        local projectedSamples = samples:select(2, k)

        -- Now, we expect the distribution of the projected samples to be mu[k], math.sqrt(sigma[k][k])
        local p, chi2 = randomkit.chi2Gaussian(projectedSamples, mu[k], math.sqrt(sigma[k][k]))

        -- Bonferroni's correction
        if p < 1 - bonferroniCorrection(alpha, D) then
            -- we're rejecting the null hypothesis, that the sample is normally distributed with the above params
            rejectionCount = rejectionCount + 1
            tester:assert(not shouldAccept, "projected sample should be accepted as gaussian with given parameters")
        end
    end

    -- If we're not supposed to be accepting this sample, check that it was rejected by at least one of the tests
    tester:assert(shouldAccept or rejectionCount > 0, "projected sample should be rejected as gaussian with given parameters")

    -- Part two: transform and chi2 test against standard normal dist'n

    -- TODO



end

local function shouldBeFromMGaussians(v1, v2, v3, desc)

    local accumulated = torch.Tensor(M, N, D):zero()

    -- Each call only returns one sample from each distribution, so to
    -- perform our statistical tests we need to make many calls.
    local notNil = true
    local correctDim = true
    local correctSize = true
    for k = 1, N do
        local results = randomkit.multivariateGaussianRand(v1, v2, v3)
        notNil = notNil and results ~= nil
        correctDim = correctDim and results:dim() == 2
        correctSize = correctSize and results:size(1) == M
        local gaussDim
        if v2:dim() == 2 then
            gaussDim = v2:size(2)
        else
            gaussDim = v2:size(1)
        end
        correctSize = correctSize and results:size(2) == gaussDim
        for j = 1, M do
            accumulated[j][k] = results[j]
        end
    end

    tester:assert(notNil, "got no result - expected samples from a gaussian!")
    tester:assert(correctDim, "wrong dimensionality for result")
    tester:assert(correctSize, "expected results of correct size")

    for j = 1, M do
        local mu = v2
        if v2:dim() == 2 then
            mu = v2[j]
        end
        local sigma = v3
        if v3:dim() == 3 then
            sigma = v3[j]
        end
        statisticalTestMultivariateGaussian(accumulated[j], mu, sigma, true)
        gnuplot.pngfigure('fig' .. j)
        gnuplot.plot('samples', accumulated[j], '.')
        gnuplot.plotflush()
    end
end

local secondArgD = torch.Tensor { 10, 0 }

local thirdArgMDD = torch.Tensor(M, D, D):zero()
for k = 1, M do
    thirdArgMDD[k] = torch.Tensor {{k+1, k}, {k, k}}
end

local bar = {}
function bar.foo()
    shouldBeFromMGaussians(M, secondArgD, thirdArgMDD, "M,D,MxDxD")
end

tester:add(bar)
tester:run()
