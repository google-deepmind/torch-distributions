require 'cephes'

local function vectorise1param(func)
    return function(x, param)
        if not randomkit._isTensor(x) and not randomkit._isTensor(param) then
            return func(x, param)
        end

        x = torch.Tensor(x)
        param = torch.Tensor(param)

        if randomkit._isTensor(x) and (not randomkit._isTensor(param) or param:size(1) == 1) then
            param = param:expand(x:nElement())
        end
        if (not randomkit._isTensor(x) or x:size(1) == 1) and randomkit._isTensor(param) then
            x = x:expand(param:nElement())
        end
        assert(x:size(1) == param:size(1))
        return x:clone():map(param, func)
    end
end

randomkit.poissonPDF = vectorise1param(function(x, lambda)
    return cephes.pow(lambda, x) / cephes.fac(x) * cephes.exp(-lambda)
end)

function randomkit.poissonLogPDF(x, lambda)
    return x * cephes.log(lambda) - cephes.lgam(x+1) - lambda
end

function randomkit.poissonCDF(x, lambda)
    if x >= 0 then
        return cephes.pdtr(x, lambda)
    else
        return 0
    end
end

function randomkit.poissonQuantile(p, lambda)
    error("Not implemented")
end
