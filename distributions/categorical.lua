require 'torch'

distributions.cat = {}

function distributions.cat.pdf(...)
    error('Not implemented')
end

function distributions.cat.logpdf(...)
    error('Not implemented')
end

local function _idxResultAndN(x, categories)
    -- return res, N
    if type(x) == 'number' then
        return torch.LongTensor(x), x
    elseif distributions._isTensor(x) then
        if categories then
            print('WARNING: cannot use result tensors and categories at the same time. Ignoring result tensor.')
            -- TODO: once index() will accept resulting tensor, use it
        end
        return x, x:numel()
    end
end

-- Categorical sampling
function distributions.cat.rnd(...)
    local I, N, p, options
    local nArgs = select('#', ...)
    if nArgs == 1 then
        -- only p
        I, N = _idxResultAndN(1)
        p = select(1, ...)
        options = {}
    elseif nArgs == 2 then
        -- (N,  p) or (p, options)
        if type(select(2, ...)) == 'table' then
            -- (p, options)
            p, options = ...
            options = options or {}
            I, N = _idxResultAndN(1, options.categories)
        else
            -- (N, p)
            I, N = _idxResultAndN(select(1, ...), nil)
            p = select(2, ...)
            options = {}
        end
    elseif nArgs == 3 then
        -- (N, p, options)
        p, options = select(2, ...)
        options = options or {}
        I, N = _idxResultAndN(select(1, ...), options.categories)
    else
        error('Expected cat.rnd([N], p, [options])')
    end

    if options.categories and options.categories:size(1) ~= p:numel() then
        error('the number of categories does not match the length of the probability vector')
    end

    -- TODO: avoid new tensor due to cumsum! See #25
    local cdf = p:cumsum(1)
    local totalmass = cdf[#cdf]
    if totalmass <= 0 then 
        error('cannot resample with total probability mass 0') 
    end
    cdf = cdf:div(totalmass)

    if not options.type or options.type == 'iid' then
        distributions.cat._iid(I, cdf)
    elseif options.type == 'dichotomy' then
        distributions.cat._dichotomy(I, cdf)
    elseif options.type == 'stratified' then
        distributions.cat._stratified(I, cdf)
    else
        error('Unknow categorical sampling type ' .. options.type)
    end

    if options.categories then
        -- TODO: once index() will accept resulting tensor, use it
        return options.categories:index(1, I)
    else
        return I
    end
end

-- IID sampling , FFI version
function distributions.cat._iid(I, cdf)
    local N = I:numel()
    local U = torch.rand(N)
    local permutation
    U, permutation = torch.sort(U)

    local udata = torch.data(U)
    local permdata = torch.data(permutation)
    local cdfdata = torch.data(cdf:contiguous())
    local idata = torch.data(I:contiguous())

    local index = 0
    for k = 0, N-1 do
        while udata[k] > cdfdata[index] do
            index = index + 1
        end
        idata[permdata[k]-1] = index + 1
    end

    return I
end

-- Categorical sampling with dichotomy search
-- Note: since randDiscrete is so much faster with FFI, 
-- it always overspeeds randDiscreteDichotomy, so better
-- use randDiscrete instead
-- Note that this FFI version is faster than a non-FFI version
-- when N > 3.
function distributions.cat._dichotomy(I, cdf)
    local N = I:numel()
    local U = torch.rand(N)
    local permutation
    U, permutation = torch.sort(U)

    local udata = torch.data(U)
    local permdata = torch.data(permutation)
    local cdfdata = torch.data(cdf:contiguous())
    local idata = torch.data(I:contiguous())
    local nBins = cdf:numel()

    local left = 0
    local right = 0
    local middle = 0
    for k = 0, N-1 do
        local d = 1
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
function distributions.cat._stratified(I, cdf)
    local N = I:numel()
    local U = torch.rand(N)

    local udata = torch.data(U)
    local cdfdata = torch.data(cdf:contiguous())
    local idata = torch.data(I:contiguous())

    local index = 0
    for k = 0, N-1 do
        while (udata[k] + k)/N > cdfdata[index] do
            index = index + 1
        end
        idata[k] = index + 1
    end

    return I
end
