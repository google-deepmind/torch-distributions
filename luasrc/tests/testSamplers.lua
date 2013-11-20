require 'torch'
require 'mathx'

local tester = torch.Tester()
local testSamplers = {}

function testSamplers.testNormalization()
    local p = torch.Tensor({0,1})
    local x = mathx.randDiscrete(p, 10)
    for i=1, 10 do
        tester:asserteq(x[i], 2, 'should be all 2')
    end

    local nfreq = 10000
    local p10 = torch.Tensor({5, 5})
    local x = mathx.randDiscrete(p10, nfreq)
    local freq={}
    for i = 1, nfreq do
        if not freq[x[i]] then freq[x[i]] = 0 end
        freq[x[i]] = freq[x[i]] + 1
    end
    tester:assert(freq[1] and freq[2], 'seems that we have a normalization problem')
end

function testSamplers.zeroprob()
    local nullProba = torch.zeros(10)
    tester:assertError(function() mathx.randDiscrete(nullProba, 1) end, 'should not be able to resample with null total mass')
end

function testSamplers.resample()
    local nrow = 30
    local ncol = 20
    local x = torch.zeros(nrow, ncol)
    for i = 1, nrow do
        x[i] = torch.range(10*i, 10*i+ncol):resize(1,ncol)
    end
    -- resample should throw an error when probabilities are not assigned to each row
    local p = torch.ones(1)
    tester:assertError(function() mathx.resample(x, p) end, 'resample() should require probabilities for each row')

    local onlySurvivor = 2
    local p = torch.zeros(nrow)
    p[onlySurvivor] = 1
    local expected = torch.Tensor(nrow,ncol)
    for i = 1, nrow do
        expected[i] = x[onlySurvivor]
    end
    tester:assertTensorEq(mathx.resample(x,p), expected, 1e-16, 'should have only survivor')

    local nExtended = 50
    local expected = torch.Tensor(nExtended,ncol)
    for i = 1, nExtended do
        expected[i] = x[onlySurvivor]
    end
    tester:assertTensorEq(mathx.resample(x,p,nExtended), expected, 1e-16, 'should have extended survivor')

    local nShrinked = 2
    local expected = torch.Tensor(nShrinked,ncol)
    for i = 1, nShrinked do
        expected[i] = x[onlySurvivor]
    end
    tester:assertTensorEq(mathx.resample(x,p,nShrinked), expected, 1e-16, 'should have extended survivor')
end

function testSamplers.testLinspace()
    -- check if the linspace bug is fixed
    tester:assertTensorEq(torch.linspace(0,0,1), torch.zeros(1), 1e-16, 'linspace bug not yet fixed, update your torch version')
end


function isSamplerSorted(sampler)
    local p = torch.ones(10)
    local nSamples = 10000
    local x = sampler(p, nSamples)
    local isSorted = true
    for i = 2,nSamples do
        if x[i] < x[i-1] then
            isSorted = false
        end
    end
    return isSorted
end

function testSamplers.testStratifiedIsSorted()
    tester:assert(isSamplerSorted(mathx.randDiscreteStratified) == true, 'Stratified indices should be sorted')
end

function testSamplers.testDichotomyIsNotSorted()
    tester:assert(isSamplerSorted(mathx.randDiscreteDichotomy) == false, 'Indices should NOT be sorted')
end

function testSamplers.testUnsortedIsNotSorted()
    tester:assert(isSamplerSorted(mathx.randDiscrete) == false, 'Indices should NOT be sorted')
end

function unbiased(sampler)
    local p = torch.ones(10)
    local nrep = 10000
    local countOne = 0
    for i = 1,nrep do
        local x = sampler(p, 1)
        if x[1] == 1 then
            countOne = countOne + 1
        end
    end
    -- Very crude hypothesis testing :p
    tester:assert(countOne < 3*nrep/10, 'Sampled 1 way too often')
end

function testSamplers.testSpeed()
    nBins = 10000
    nSamples = 2
    p = torch.ones(nBins)
    timer = torch.Timer()
    x = mathx.randDiscrete(p, nSamples)
    elapsedDiscrete = timer:time().real
    x = mathx.randDiscreteDichotomy(p, nSamples)
    elapsedDichotomy = timer:time().real - elapsedDiscrete
    print('Dichotomy: ' .. elapsedDichotomy .. ' VS Linear: ' .. elapsedDiscrete .. ', i.e. factor ' .. elapsedDiscrete/elapsedDichotomy)
    tester:assert(elapsedDiscrete > elapsedDichotomy, 'Naive linear search is faster than dichotomic !')
end

function testSamplers.testUnsortedWithOneSample()
    unbiased(mathx.randDiscrete)
end

function testSamplers.testDichotomyWithOneSample()
    unbiased(mathx.randDiscreteDichotomy)
end

function testSamplers.testStratifiedWithOneSample()
    unbiased(mathx.randDiscreteStratified)
end

tester:add(testSamplers)
tester:run()
