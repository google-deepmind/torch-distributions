require 'cephes'

distributions.poisson = {}

local function vectorise1param(func)
    return function(x, param)
        if not distributions._isTensor(x) and not distributions._isTensor(param) then
            return func(x, param)
        end

        x = torch.Tensor(x)
        param = torch.Tensor(param)

        if distributions._isTensor(x) and (not distributions._isTensor(param) or param:size(1) == 1) then
            param = param:expand(x:nElement())
        end
        if (not distributions._isTensor(x) or x:size(1) == 1) and distributions._isTensor(param) then
            x = x:expand(param:nElement())
        end
        assert(x:size(1) == param:size(1))
        return x:clone():map(param, func)
    end
end

distributions.poisson.pdf = vectorise1param(function(x, lambda)
    return cephes.pow(lambda, x) / cephes.fac(x) * cephes.exp(-lambda)
end)

function distributions.poisson.logpdf(x, lambda)
    return x * cephes.log(lambda) - cephes.lgam(x+1) - lambda
end

function distributions.poisson.cdf(x, lambda)
    if x >= 0 then
        return cephes.pdtr(x, lambda)
    else
        return 0
    end
end

function distributions.poisson.qtl(p, lambda)
    error("Not implemented")
end

function distributions.poisson.rnd(p, lambda)
    error("Not implemented")
end
