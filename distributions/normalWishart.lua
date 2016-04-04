-- TODO: implement in-place version of functions
distributions.nw = {}

--[[ Log probability density function of a mean and precision under a
    Normal-Wishart distribution

Parameters:

* `mean` (1D Tensor) data - mean vector
* `precision` (2D Tensor) data - precision matrix
* `loc` (1D Tensor) location
* `beta` (number>0) scales the precision matrix which we use to sample the mean
* `scale` (2D Tensor) scale matrix for Wishart distribution
* `ndof` (number) degrees of freedom for the Wishart
    Must be greater than the dimension of the data minus one

Returns:

1. log probability density
]]
function distributions.nw.logpdf(mean, precision, loc, beta, scale, ndof)
  local ndim = mean:size(1)
  assert(precision:size(1) == ndim)
  assert(precision:size(2) == ndim)
  assert(loc:size(1) == ndim)
  assert(scale:size(1) == ndim)
  assert(scale:size(2) == ndim)
  assert(ndof > ndim - 1)
  return distributions.mvn.logpdf(mean, loc, torch.inverse(precision * beta))
      + distributions.wishart.logpdf(precision, ndof, scale)
end

--[[ Probability density function of a mean and precision under a
    Normal-Wishart distribution

Parameters:

* `mean` (1D Tensor) data - mean vector
* `precision` (2D Tensor) data - precision matrix
* `loc` (1D Tensor) location
* `beta` (number>0) scales the precision matrix which we use to sample the mean
* `scale` (2D Tensor) scale matrix for Wishart distribution
* `ndof` (number) degrees of freedom for the Wishart
    Must be greater than the dimension of the data minus one

Returns:

1. Probability density
]]
function distributions.nw.pdf(...)
  return cephes.exp(distributions.nw.logpdf(...))
end

--[[ Sample mean and precision from a Normal-Wishart distribution

Parameters:

* `loc` (1D Tensor) location
* `beta` (number>0) scales the precision matrix which we use to sample the mean
* `scale` (2D Tensor) scale matrix for Wishart distribution
* `ndof` (number>scale:size() - 1) degrees of freedom for the Wishart

Returns:

1. mean (1D Tensor)
2. precision (2D Tensor)
]]
function distributions.nw.rnd(loc, beta, scale, ndof)
  local precision = distributions.wishart.rnd(ndof, scale:size(1), scale)
  local mean = distributions.mvn.rnd(loc, torch.inverse(
                                                 precision * beta):contiguous())
  return mean, precision
end

function distributions.nw.entropy(loc, beta, scale, ndof)
  local ndim = loc:size(1)
  assert(scale:size(1) == ndim)
  return ndim * (1 + torch.log(2*math.pi)) / 2
      - ndim * torch.log(beta) / 2
      - distributions.wishart._elogdet(ndof, ndim, scale) / 2
      + distributions.wishart.entropy(ndof, scale)
end

function distributions.nw.kl(q, p)
  local ndim = q.loc:size(1)
  local function qf(A, x) return torch.dot(x, torch.mv(A,x)) end
  return distributions.wishart.kl(q, p)
      + (ndim * (math.log(q.beta) - math.log(p.beta))
      + ndim * p.beta / q.beta
      - ndim
      + p.beta * q.ndof
      * qf(q.scale, p.loc - q.loc)) / 2
end
