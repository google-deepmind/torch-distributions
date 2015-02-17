require 'totem'
require 'distributions'

local mytest = {}
local tester = totem.Tester()

function mytest.testKSGaussian()
    local nPoints = 1000
    local mu = 5
    local sigma = 10
    local x, y  = torch.randn(nPoints), torch.randn(nPoints)
    x = x * sigma + mu
    y = y * sigma + mu
    -- KS test at 99%
    local p, d = distributions.kstwo(x, y)
    tester:assert(d, 'Test statistic should not be nil')
    tester:assert(p > 0.01, 'KS test rejects two samples from a same gaussian distribution')
    x = x + 2*mu
    local p, d = distributions.kstwo(x, y)
    tester:assert(d, 'Test statistic should not be nil')
    tester:assert(p < 0.01, 'KS test accepts two samples from different gaussian distributions')
end


function mytest.testKSOne()
    local nPoints = 1000
    local mu = 5
    local sigma = 10
    local x, y = torch.randn(nPoints)
    x = x * sigma + mu
    local function cdf(x)
      return distributions.norm.cdf(x, mu, sigma)
    end
    -- KS test at 99%
    local p, d = distributions.ksone(x, cdf)
    tester:assert(d, 'Test statistic should not be nil')
    tester:assert(p > 0.01, 'KS test rejects a sample from a gaussian distribution')
    x = x + 2*mu
    local p, d = distributions.ksone(x, cdf)
    tester:assert(d, 'Test statistic should not be nil')
    tester:assert(p < 0.01, 'KS test accepts a sample from a different gaussian distribution')
end

tester:add(mytest)
return tester:run()
