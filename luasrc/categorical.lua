--Methods for sampling and resampling particles and multinomials
require 'torchffi'

-- TODO: support result tensor
distributions.cat = {}

function distributions.cat.pdf(...)
    error('Not implemented')
end

function distributions.cat.logpdf(...)
    error('Not implemented')
end

-- Multinomial sampling
function distributions.cat.rnd(...)
    local N, p, options
    local nArgs = select('#', ...)
    if nArgs == 1 then
        -- only p
        N = 1
        p = ...
        options = {type = 'iid'}
    elseif nArgs == 2 then
        -- (N,  p) or (p, options)
        if type(select(2, ...)) == 'table' then
            N = 1
            p, options = ...
        else
            N, p = ...
        end
    elseif nArgs == 3 then
        -- (N, p, options)
        N, p, options = ...
    else
        error('Expected cat.rnd([N], p, [options])')
    end

    local cdf = p:cumsum(1)
    local totalmass = cdf[#cdf]
    if totalmass <= 0 then 
        error('cannot resample with total probability mass 0') 
    end
    cdf = cdf:div(totalmass)

    if not options or not options.type or options.type == 'iid' then
        return distributions.cat._iid(N, cdf)
    elseif options.type == 'dichotomy' then
        return distributions.cat._dichotomy(N, cdf)
    elseif options.type == 'stratified' then
        return distributions.cat._stratified(N, cdf)
    else
        error('Unknow categorical sampling type ' .. options.type)
    end
end

-- IID sampling , FFI version
function distributions.cat._iid(N, cdf)
    local I = torch.LongTensor(N)
    local U = torch.rand(N)
    local permutation
    U, permutation = torch.sort(U)

    local udata = torch.data(U)
    local permdata = torch.data(permutation)
    local cdfdata = torch.data(cdf:contiguous())
    local idata = torch.data(I)

    local index = 0
    for k = 0, N-1 do
        while udata[k] > cdfdata[index] do
            index = index + 1
        end
        idata[permdata[k]-1] = index + 1
    end

    return I
end

-- Multinomial sampling with dichotomy search
-- Note: since randDiscrete is so much faster with FFI, 
-- it always overspeeds randDiscreteDichotomy, so better
-- use randDiscrete instead
-- Note that this FFI version is faster than a non-FFI version
-- when N > 3.
function distributions.cat._dichotomy(N, cdf)
    local I = torch.LongTensor(N)
    local U = torch.rand(N)
    local permutation
    U, permutation = torch.sort(U)

    local udata = torch.data(U)
    local permdata = torch.data(permutation)
    local cdfdata = torch.data(cdf:contiguous())
    local idata = torch.data(I)

    local left = 0
    local right = 0
    local middle = 0
    for k = 0, N-1 do
        d = 1
        while udata[k] > cdfdata[right] do
            left = right
            right = math.min(right + 2^d, nBins-1)
            d = d + 1
        end
        while right - left > 1 do
            middle = math.floor(left + right)/2
            if udata[k] > cdfdata[middle] then
                left = middle
            else
                right = middle
            end
        end
        idata[permdata[k]-1] = right+1
    end

    return I
end

-- sorted stratified sampler
function distributions.cat._stratified(N, cdf)
    local I = torch.LongTensor(N)
    local U = torch.rand(N)

    local udata = torch.data(U)
    local cdfdata = torch.data(cdf:contiguous())
    local idata = torch.data(I)

    local index = 0
    for k = 0, N-1 do
        while (udata[k] + k)/N > cdfdata[index] do
            index = index + 1
        end
        idata[k] = index + 1
    end

    return I
end
