-- TODO: implement in-place version of functions
require 'randomkit'
require 'cephes'
require 'torch'

if not distributions.gamma then
  distributions.gamma = {}
end

function distributions.gamma.entropy(params)
  assert(params.shape)
  assert(params.scale or params.rate)

  if params.scale then
    local k = params.shape
    local t = params.scale
    return k + math.log(t) + cephes.lgam(k) + (1-k)*cephes.digamma(k)
  elseif params.rate then
    local a = params.shape
    local b = params.rate
    return a - math.log(b) + cephes.lgam(a) + (1-a)*cephes.digamma(a)
  end
end

function distributions.gamma.kl(q,p)
  if q.scale then
    if p.scale then
      return (q.shape - p.shape) * cephes.digamma(q.shape)
          - q.shape
          + q.shape * q.scale / p.scale
          + cephes.lgam(p.shape)
          - cephes.lgam(q.shape)
          + p.shape * math.log(p.scale)
          - p.shape * math.log(q.scale)
    elseif p.rate then
      return (q.shape - p.shape) * cephes.digamma(q.shape)
          - q.shape
          + q.shape * q.scale * p.rate
          + cephes.lgam(p.shape)
          - cephes.lgam(q.shape)
          - p.shape * math.log(p.rate)
          - p.shape * math.log(q.scale)
    end
  elseif q.rate then
    if p.scale then
      return (q.shape - p.shape) * cephes.digamma(q.shape)
          - q.shape
          + q.shape / q.rate / p.scale
          + cephes.lgam(p.shape)
          - cephes.lgam(q.shape)
          + p.shape * math.log(p.scale)
          + p.shape * math.log(q.rate)
    elseif p.rate then
      return (q.shape - p.shape) * cephes.digamma(q.shape)
          - q.shape
          + q.shape * p.rate / q.rate
          + cephes.lgam(p.shape)
          - cephes.lgam(q.shape)
          - p.shape * math.log(p.rate)
          + p.shape * math.log(q.rate)
    end
  end
end