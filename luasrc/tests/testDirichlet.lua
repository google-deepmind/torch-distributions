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

function myTests.testDirichletEntropy()
  local alpha = torch.Tensor({0.2,1.0,3.0})
  tester:assert(distributions.dir.entropy(alpha))
end

function myTests.testDirichletKL()
  local alpha_p = torch.Tensor({0.2,1.0,3.0})
  local alpha_q = torch.Tensor({0.6,0.2,1.5})
  tester:assert(distributions.dir.kl(alpha_p,alpha_q) >= 0)
  tester:assert(distributions.dir.kl(alpha_q,alpha_p) >= 0)
  tester:asserteq(distributions.dir.kl(alpha_p,alpha_p), 0)
  tester:asserteq(distributions.dir.kl(alpha_q,alpha_q), 0)
end

tester:add(myTests)
return tester:run()