require 'totem'
require 'distributions'
require 'torch'
require 'pl.strict'

local myTests = {}
local tester = totem.Tester()
torch.manualSeed(1234567890)

function myTests.testWishartPdf()
  local D = 5
  local nu = D + 10
  local V = torch.randn(D,D)
  V = V * V:t()

  local prec = torch.randn(D,D)
  prec = prec * prec:t()

  tester:assert(distributions.wishart.pdf(prec, nu, D, V))
  tester:assert(distributions.wishart.logpdf(prec, nu, D, V))
end

function myTests.testWishartRnd()
  local D = 5
  local nu = D + 10
  local V = torch.randn(D,D)
  V = V * V:t()

  tester:assert(distributions.wishart.rnd(nu, D, V))
end

tester:add(myTests)
return tester:run()