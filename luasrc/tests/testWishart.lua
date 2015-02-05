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

function myTests.testWishartEntropy()
  local D = 5
  local nu = D + 10
  local V = torch.randn(D,D)
  V = V * V:t()

  tester:assert(distributions.wishart.entropy(nu, V))
end

function myTests.testWishartKL()
  local D = 5

  local p = {}
  local q = {}

  p.ndof = D + 10
  q.ndof = D + 5

  p.scale = torch.randn(D,D)
  p.scale = V * V:t()

  q.scale = torch.randn(D,D)
  q.scale = torch.randn(D,D)

  tester:assert(distributions.wishart.kl(p,q))

  q.inverseScale = torch.inverse(q.scale)
  q.scale = nil
  tester:assert(distributions.wishart.kl(p,q))
end

tester:add(myTests)
return tester:run()