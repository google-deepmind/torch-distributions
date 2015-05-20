require 'torch'
require 'strict'
local dist = require('distributions')

local nBins = {1000, 10000, 100000, 1000000}
local nSamples = {1, 10, 100, 1000}

local timer = torch.Timer()

for _, bins in ipairs(nBins) do
  local p = torch.ones(bins)
  for _, samples in ipairs(nSamples) do
    local start = timer:time().real
    for i = 1, 100 do
      dist.cat.rnd(samples, p)
    end
    elapsedDiscrete = timer:time().real - start
    collectgarbage()

    start = timer:time().real
    for i = 1, 100 do
      dist.cat.rnd(samples, p, {type = 'dichotomy'})
    end
    elapsedDichotomy = timer:time().real - start
    collectgarbage()

    local conclusion
    if elapsedDiscrete < elapsedDichotomy then
      conclusion = string.format('Linear faster than dichotomy by x%0.3f',
        elapsedDiscrete/elapsedDichotomy)
    else
      conclusion = string.format('Dichotomy faster than linear by x%0.3f',
        elapsedDiscrete/elapsedDichotomy)
    end
    print(string.format('%4d samples, %7d bins (x100): %0.4fs (linear) ' ..
      'vs %0.4fs (dichotomic). %s', samples, bins, elapsedDiscrete, elapsedDichotomy, conclusion))
  end
end
