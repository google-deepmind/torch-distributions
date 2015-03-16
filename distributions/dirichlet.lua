-- TODO: implement in-place version of functions
require 'torch'
require 'cephes'
require 'randomkit'
distributions.dir = {}

-- Return a tensor normalized to sum to one
local function _normalize(x)
  return x / x:sum()
end

--[[ Generate a sample from a Dirichlet distribution

Parameters:

* `alpha` (Tensor or table) vector of pseudocounts

Returns:

1. sample from a Dirichlet distribution (Tensor)
]]
function distributions.dir.rnd(alpha)
  if type(alpha) == 'table' then alpha = torch.Tensor(alpha) end
  return _normalize(randomkit.gamma(alpha, torch.Tensor(alpha:size()):fill(1)))
end

--[[ Probability density of a multinomial distribution
    under a Dirichlet distribution

Parameters:

* `x` (Tensor) multinomial distribution
    entries must be nonnegative and sum to one
* `alpha` (Tensor) vector of pseudocounts

Returns:

1. Probability density
]]
function distributions.dir.pdf(...)
  return cephes.exp(distributions.dir.logpdf(...))
end

--[[ Log probability density of a multinomial distribution
    under a Dirichlet distribution

Parameters:

* `x` (Tensor) multinomial distribution
    entries must be nonnegative and sum to one
* `alpha` (Tensor) vector of pseudocounts

Returns:

1. log probability density
]]
function distributions.dir.logpdf(x, alpha)
  assert(x:nElement() == alpha:nElement())
  return torch.log(x):cmul(alpha-1):sum() 
      - cephes.lgam(alpha):sum() 
      + cephes.lgam(torch.sum(alpha))
end

-- returns the entropy of a Dirichlet distribution from a vector of parameters
function distributions.dir.entropy(alpha)
  local alpha0 = alpha:sum()
  return cephes.lgam(alpha):sum() - cephes.lgam(alpha0) 
      + (alpha0-alpha:nElement())*cephes.digamma(alpha0)
      - cephes.digamma(alpha):cmul(alpha-1):sum()
end

-- returns the KL divergence KL[q || p] 
-- q and p are both tables with field 'alpha', a vector of pseudocounts
function distributions.dir.kl(q, p)
  local q0 = q.alpha:sum()
  local p0 = p.alpha:sum()
  return cephes.lgam(q0) - cephes.lgam(p0)
      - cephes.lgam(q.alpha):sum() + cephes.lgam(p.alpha):sum()
      + (p0 - q0) * cephes.digamma(q0)
      + cephes.digamma(q.alpha):cmul(q.alpha - p.alpha):sum()
end
