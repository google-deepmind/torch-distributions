-- TODO: implement in-place version of functions
require 'randomkit'
require 'cephes'
require 'torch'

distributions.wishart = {}

--[[ Generate a sample from a Wishart distribution

Parameters:

* `ndof` (number) Degrees of freedom
* `ndim` (number) Dimension of output matrix
* `scale` (optional 2D Tensor) Scale matrix.
    Must be positive definite.
    Defaults to the identity if not provided.

Returns:

1. positive definite matrix sampled from Wishart distribution (Tensor)
]]
function distributions.wishart.rnd(ndof, ndim, scale)
  assert(ndof > ndim - 1)
  if scale then
    assert(scale:size(1) == scale:size(2))
    assert(scale:size(1) == ndim)
    assert(distributions.util.isposdef(scale))
  end

  --[[ Sample from a Wishart distribution with identity scale matrix
  Uses the Bartlett decomposition form of the algorithm of 
  Odell and Feiveson (1966), described in section 3 of this paper:
  http://www.math.wustl.edu/~sawyer/hmhandouts/Wishart.pdf
  ]]
  local T = torch.zeros(ndim, ndim)
  for i = 1, ndim do
    T[i][i] = torch.sqrt(randomkit.chisquare(ndof-i+1))
    for j = i+1, ndim do
      T[i][j] = torch.normal()
    end
  end

  if scale then
      local cholScale = torch.potrf(scale)
      return cholScale:t() * T * T:t() * cholScale
  else
      return T * T:t()
  end
end

-- Log normalizer for the Wishart distribution
function distributions.wishart._lognorm(ndof, ndim, scale)
  return ndof * ndim * cephes.log(2) / 2
      + ndof * distributions.util.logdet(scale) / 2
      + cephes.lmvgam(ndof/2, ndim)
end

-- Log normalizer for the Wishart distribution, with inverse scale
function distributions.wishart._lognorm2(ndof, ndim, invScale)
  return ndof * ndim * cephes.log(2) / 2
      - ndof * distributions.util.logdet(invScale) / 2
      + cephes.lmvgam(ndof/2, ndim)
end

-- Expectation of the log determinant of an observation
function distributions.wishart._elogdet(ndof, ndim, scale)
  return ndim * cephes.log(2)
      + distributions.util.logdet(scale)
      + cephes.digamma((-torch.range(1,ndim) + ndof + 1)/2):sum()
end

--[[ Log probability density of a positive definite matrix
    under a Wishart distribution

Parameters:

* `X` (Tensor) Data. Must be positive definite
* `ndof` (number) Degrees of freedom of Wishart distribution
* `scale` (Tensor) Scale matrix of Wishart distribution. 
    Must be positive definite.

Returns:

1. log probability density p(X | dof, scale)
]]
function distributions.wishart.logpdf(X, ndof, scale)
  local ndim = scale:size(1)
  assert(scale:size(1) == scale:size(2))
  assert(distributions.util.isposdef(scale))

  assert(ndof > ndim - 1)

  assert(X:size(1) == ndim)
  assert(X:size(2) == ndim)
  assert(distributions.util.isposdef(X))

  return (ndof - ndim - 1) * distributions.util.logdet(X) / 2
      - torch.dot(torch.inverse(scale), X) / 2
      - distributions.wishart._lognorm(ndof, ndim, scale)
end

--[[ Probability density of a positive definite matrix
    under a Wishart distribution

Parameters:

* `X` (Tensor) Data. Must be positive definite
* `ndof` (number) Degrees of freedom of Wishart distribution
* `scale` (Tensor) Scale matrix of Wishart distribution. 
    Must be positive definite.

Returns:

1. Probability density p(X | dof, scale)
]]
function distributions.wishart.pdf(...)
  return cephes.exp(distributions.wishart.logpdf(...))
end

function distributions.wishart.entropy(ndof, scale)
  local ndim = scale:size(1)
  assert(distributions.util.isposdef(scale))

  return distributions.wishart._lognorm(ndof, ndim, scale)
      - (ndof - ndim - 1) distributions.wishart._elogdet(ndof, ndim, scale) / 2
      + (ndim * ndof) / 2

  -- return cephes.lmvgam(ndof/2, ndim)
  --     + (ndim + 1) * distributions.util.logdet(scale) / 2
  --     + ndim * (ndim + 1) * cephes.log(2) / 2
  --     - (ndof - ndim - 1) 
  --     * cephes.digamma((-torch.range(1,ndim) + ndof + 1) / 2):sum() / 2
  --     + ndim * ndof / 2
end

-- KL divergence between two Wishart distributions
-- The computation is fastest if we use the scale for
-- p and inverse scale for q, but works if either the
-- field 'scale' or 'inverseScale' is provided with q.
function distributions.wishart.kl(params_p, params_q)
  local ndim = params_p.scale:size(1)
  local ndof_p = params_p.ndof
  local ndof_q = params_q.ndof

  local scale_p = params_p.scale
  local invScale_q
  if params_q.inverseScale then
    invScale_q = params_q.inverseScale
  else
    invScale_q = torch.inverse(params_q.scale)
  end

  return (ndof_p - ndof_q) * 
      distributions.wishart._elogdet(ndof_p, ndim, scale_p)
      - ndof_p * ndim / 2
      + ndof_p * torch.dot(invScale_q, scale_p)
      + distributions.wishart._lognorm2(ndof_q, ndim, invScale_q)
      - distributions.wishart._lognorm(ndof_p, ndim, scale_p)

  -- return (ndof_p - ndof_q) 
  --     * cephes.digamma((-torch.range(1, ndim) + ndof_p + 1)/2):sum() / 2
  --     - ndof_q * distributions.util.logdet(scale_p) / 2
  --     - ndof_q * distributions.util.logdet(invScale_q) / 2
  --     - ndof_p * ndim / 2
  --     + ndof_p * torch.dot(invScale_q, scale_p) / 2
  --     + cephes.lmvgam(ndof_q/2, ndim)
  --     - cephes.lmvgam(ndof_p/2, ndim)
end