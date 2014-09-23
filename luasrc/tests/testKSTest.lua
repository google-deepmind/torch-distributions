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
    -- Chi-square test at 99%
    local p, d = distributions.kstwo(x, y)
    tester:assert(d, 'Test statistic should not be nil')
    tester:assert(p > 0.01, 'KS test rejects two samples from a same gaussian distribution')
    x = x + 2*mu
    local p, d = distributions.kstwo(x, y)
    tester:assert(d, 'Test statistic should not be nil')
    tester:assert(p < 0.01, 'KS test accepts two samples from different gaussian distributions')
end


tester:add(mytest)
return tester:run()
