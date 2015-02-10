require 'totem'
require 'distributions'
require 'torch'
require 'pl.strict'

local myTests = {}
local tester = totem.Tester()
torch.manualSeed(1234567890)

function myTests.testNormalWishartRnd()
  local D = 5
  local loc = torch.randn(D)
  local beta = 10
  local nu = D + 10
  local V = torch.randn(D,D)
  V = V * V:t()

  tester:assert(distributions.nw.rnd(loc, beta, V, nu))
end

function myTests.testNormalWishartPdf()
  local D = 5
  local loc = torch.randn(D)
  local beta = 10
  local nu = D + 10
  local V = torch.randn(D,D)
  V = V * V:t()

  local mean = torch.randn(D)
  local prec = torch.randn(D,D)
  prec = prec * prec:t()

  tester:assert(distributions.nw.pdf(mean, prec, loc, beta, V, nu))
  tester:assert(distributions.nw.logpdf(mean, prec, loc, beta, V, nu))
end

function myTests.testNormalWishartEntropy()
  local D = 5
  local loc = torch.randn(D)
  local beta = 10
  local nu = D + 10
  local V = torch.randn(D,D)
  V = V * V:t()

  tester:assert(distributions.nw.entropy(loc, beta, V, nu))
end

function myTests.testNormalWishartKL()
  local D = 5

  local p = {}
  local q = {}

  p.loc = torch.randn(D)
  p.ndof = D + 5
  p.scale = torch.randn(D,D)
  p.scale = p.scale * p.scale:t()
  p.beta = 3

  q.loc = torch.randn(D)
  q.ndof = D + 10
  q.scale = torch.randn(D,D)
  q.scale = q.scale * q.scale:t()
  q.beta = 5

  tester:assert(distributions.nw.kl(p,q) >= 0)
  tester:assert(distributions.nw.kl(q,p) >= 0)
  tester:assertalmosteq(distributions.nw.kl(p,p), 0, 1e-12)
  tester:assertalmosteq(distributions.nw.kl(q,q), 0, 1e-12)
end

tester:add(myTests)
return tester:run()