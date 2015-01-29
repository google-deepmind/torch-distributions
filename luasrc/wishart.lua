require 'randomkit'
require 'cephes'
require 'torch'
require 'mathx'

distributions.wishart = {}

--[[ Generate a sample from a Wishart distribution

Parameters:

* `ndof` (number) Degrees of freedom
* `ndim` (number) Dimension of output matrix
* `scale` (optional Tensor) Scale matrix.
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
    assert(mathx.isposdef(scale))
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
  assert(mathx.isposdef(scale))

  assert(ndof > ndim - 1)

  assert(X:size(1) == ndim)
  assert(X:size(2) == ndim)
  assert(mathx.isposdef(X))

  --[[ Log Multivariate Gamma Function.
      The multivariate Gamma function generalizes the gamma function:
      $\Gamma_p(x) = \pi^{p(p-1)/4} \prod_{j=1}^p \Gamma[x + (1-j)/2]

  Parameters:

  * `x` value passed to multivariate gamma function
  * `p` degree of multivariate gamma function. If 1, reduces to log gamma

  Returns:
   
  1. log($\Gamma_p(x)$)
  ]]
  local function _lmvgam(x,p)
    local result = p * (p-1) / 4 * math.log(math.pi)
    for j = 1, p do
      result = result + cephes.lgam(x + (1-j)/2)
    end
    return result
  end

  return (ndof - ndim - 1) * mathx.logdet(X) / 2
      - (torch.inverse(scale) * X):trace()/2
      - (ndof * ndim) * torch.log(2) / 2
      - ndof * mathx.logdet(scale) / 2
      - _lmvgam(ndof/2, ndim)
end