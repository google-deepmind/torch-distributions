require 'torchffi'

distributions.mvcat = {}

function distributions.mvcat.pdf(...)
    error('Not implemented')
end

function distributions.mvcat.logpdf(...)
    error('Not implemented')
end

local function _idxResultAndN(x, p, options)
    -- return res, N
    local N, K, R
    if p:dim() ~= 2 then
        error('Mvcat requires a matrix of probabilities')
    end
    -- R rows, each with K categories
    R = p:size(1)
    K = p:size(2)

    if options and options.categories then
        error('Mvcat does not support categories, can only return integers')
    end

    if type(x) == 'number' then
        -- Given a number of results
        N = x
        return torch.LongTensor(R, N), N
    elseif distributions._isTensor(x) then
        -- Given a result tensor
        if x:dim() ~= 2 then
            error('Mvcat requires a matrix as a result tensor')
        end
        if x:size(1) ~= R then
            error('Mvcat result tensor must match number of rows in probability maitrx')
        end
        N = x:size(2)
        return x, N
    end
end

-- Categorical sampling for multiple laws at one
function distributions.mvcat.rnd(...)
    local res, N, p, options, K, D
    local nArgs = select('#', ...)
    if nArgs == 1 then
        -- only p
        p = select(1, ...)
        I, N = _idxResultAndN(1, p)
        options = {}
    elseif nArgs == 2 then
        -- (N,  p) or (p, options)
        if type(select(2, ...)) == 'table' then
            -- (p, options)
            p, options = ...
            options = options or {}
            I, N = _idxResultAndN(1, p, options)
        else
            -- (N, p)
            p = select(2, ...)
            I, N = _idxResultAndN(select(1, ...), p)
            options = {}
        end
    elseif nArgs == 3 then
        -- (N, p, options)
        p, options = select(2, ...)
        options = options or {}
        I, N = _idxResultAndN(select(1, ...), p, options.categories)
    else
        error('Expected mvcat.rnd([N], p, [options])')
    end

    for r = 1, p:size(1) do
        -- Process each row in turn
        distributions.cat.rnd(I:select(1, r), p:select(1, r), options)
    end

    return I
end
