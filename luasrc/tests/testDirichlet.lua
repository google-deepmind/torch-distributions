require 'totem'
require 'distributions'
require 'torch'
require 'pl.strict'

local myTests = {}
local tester = totem.Tester()
torch.manualSeed(1234567890)

function myTests.testDirichletRnd()
  local alpha = torch.Tensor({0.2,1.0,3.0})
  tester:assert(distributions.dir.rnd(alpha))
end

function myTests.testDirichletPdf()
  local alpha = torch.Tensor({0.2,1.0,3.0})
  local pi = torch.Tensor({0.1,0.3,0.6})
  tester:assert(distributions.dir.pdf(pi, alpha))
  tester:assert(distributions.dir.logpdf(pi, alpha))
end

tester:add(myTests)
return tester:run()