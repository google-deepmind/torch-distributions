require 'torch'
require 'cephes'
require 'randomkit'
distributions.dirichlet = {}

--[[ Generate a sample from a Dirichlet distribution

Parameters:

* `alpha` (Tensor or table) vector of pseudocounts

Returns:

1. sample from a Dirichlet distribution (Tensor)
]]
function distributions.dirichlet.rnd(alpha)
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
function distributions.dirichlet.pdf(x, alpha)
  return cephes.exp(distributions.dirichlet.logpdf(x, alpha))

--[[ Log probability density of a multinomial distribution
    under a Dirichlet distribution

Parameters:

* `x` (Tensor) multinomial distribution
    entries must be nonnegative and sum to one
* `alpha` (Tensor) vector of pseudocounts

Returns:

1. log probability density
]]
function distributions.dirichlet.logpdf(x, alpha)
  assert(x:nElement() == alpha:nElement())
  return torch.log(x):cmul(alpha-1):sum() 
      - cephes.lgam(alpha):sum() 
      + cephes.lgam(torch.sum(alpha))
end