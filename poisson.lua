require 'cephes'

local function isTensor(v)
    if torch.typename(v) then
        return string.sub(torch.typename(v), -6, -1) == "Tensor"
    end
end

local function vectorise1param(func)
    return function(x, param)
        if not isTensor(x) and not isTensor(param) then
            return func(x, param)
        end

        x = torch.Tensor(x)
        param = torch.Tensor(param)

        if isTensor(x) and (not isTensor(param) or param:size(1) == 1) then
            param = param:expand(x:nElement())
        end
        if (not isTensor(x) or x:size(1) == 1) and isTensor(param) then
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
