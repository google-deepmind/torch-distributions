-- TODO: implement in-place version of functions
distributions.nw = {}

--[[ Log probability density function of a mean and precision under a
    Normal-Wishart distribution

Parameters:

* `mean` (1D Tensor) data - mean vector
* `precision` (2D Tensor) data - precision matrix
* `loc` (1D Tensor) location
* `lam` (number>0) scales the precision matrix which we use to sample the mean
* `scale` (2D Tensor) scale matrix for Wishart distribution
* `ndof` (number) degrees of freedom for the Wishart
    Must be greater than the dimension of the data minus one

Returns:

1. log probability density
]]
function distributions.nw.logpdf(mean, precision, loc, lam, scale, ndof)
  local ndim = mean:size(1)
  assert(precision:size(1) == ndim)
  assert(precision:size(2) == ndim)
  assert(loc:size(1) == ndim)
  assert(scale:size(1) == ndim)
  assert(scale:size(2) == ndim)
  assert(ndof > ndim - 1)
  return distributions.mvn.logpdf(mean, loc, torch.inverse(precision * lam))
      + distributions.wishart.logpdf(precision, ndof, scale)
end

--[[ Probability density function of a mean and precision under a
    Normal-Wishart distribution

Parameters:

* `mean` (1D Tensor) data - mean vector
* `precision` (2D Tensor) data - precision matrix
* `loc` (1D Tensor) location
* `lam` (number>0) scales the precision matrix which we use to sample the mean
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
* `lam` (number>0) scales the precision matrix which we use to sample the mean
* `scale` (2D Tensor) scale matrix for Wishart distribution
* `ndof` (number>scale:size() - 1) degrees of freedom for the Wishart

Returns:

1. mean (1D Tensor)
2. precision (2D Tensor)
]]
function distributions.nw.rnd(loc, lam, scale, ndof)
  local precision = distributions.wishart.rnd(ndof, scale:size(1), scale)
  local mean = distributions.mvn.rnd(loc, torch.inverse(precision * lam))
  return mean, precision
end

function distributions.nw.entropy(loc, lam, scale, ndof)
  return distributions.mvn.entropy(torch.inverse(scale * lam))
      + distributions.wishart.entropy(ndof, scale)
end

function distributions.nw.kl(params_p, params_q)
  return 
      distributions.mvn.kl({
          mu = params_p.loc, 
          sigma = torch.inverse(params_p.scale * params_p.lam)
      }, 
      {
          mu = params_q.loc,
          sigma = torch.inverse(params_q.scale * params_q.lam)
          lambda = params_q.scale * params_q.lam
      }) 
      + distributions.wishart.kl(params_p, params_q)
end