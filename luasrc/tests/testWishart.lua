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

  tester:assert(distributions.wishart.pdf(prec, nu, V))
  tester:assert(distributions.wishart.logpdf(prec, nu, V))
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
  p.scale = p.scale * p.scale:t()

  q.scale = torch.randn(D,D)
  q.scale = q.scale * q.scale:t()

  tester:assert(distributions.wishart.kl(p,q) >= 0)
  tester:assert(distributions.wishart.kl(q,p) >= 0)
  tester:assertalmosteq(distributions.wishart.kl(p,p), 0, 1e-12)
  tester:assertalmosteq(distributions.wishart.kl(q,q), 0, 1e-12)

  q.inverseScale = torch.inverse(q.scale)
  q.scale = nil
  tester:assert(distributions.wishart.kl(p,q) >= 0)
end

tester:add(myTests)
return tester:run()