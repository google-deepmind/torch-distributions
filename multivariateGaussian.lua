function randomkit.multivariateGaussianLogPDF(x, mu, sigma)
    x = torch.Tensor(x)
    mu = torch.Tensor(mu)

    -- If any input is vectorised, we return a vector. Otherwise remember that we should return scalar.
    local scalarResult = (x:dim() == 1) and (mu:dim() == 1)

    -- Now make our inputs all vectors, for simplicity
    if x:dim() == 1 then
        x:resize(1, x:nElement())
    end
    if mu:dim() == 1 then
        mu:resize(1, mu:nElement())
    end

    -- Expand any 1-row inputs so that we have matching sizes
    local nResults
    if x:size(1) == 1 and mu:size(1) ~= 1 then
        nResults = mu:size(1)
        x = x:expand(nResults, x:size(2))
    elseif x:size(1) ~= 1 and mu:size(1) == 1 then
        nResults = x:size(1)
        mu = mu:expand(nResults, mu:size(2))
    else
        if x:size(1) ~= mu:size(1) then
            error("x and mu should have the same number of rows")
        end
        nResults = x:size(1)
    end

    x = x:clone():add(-1, mu)

    local logdet
    local transformed

    -- For a diagonal covariance matrix, we allow passing a vector of the diagonal entries
    if sigma:dim() == 1 then
        local D = sigma:size(1)
        local decomposed = sigma:sqrt()
        logdet = decomposed:clone():log():sum()
        transformed = torch.cdiv(x, decomposed:resize(1, D):expand(nResults, D))
    else
        local decomposed = torch.potrf(sigma):triu() -- TODO remove triu as torch will be fixed
        transformed = torch.mm(x, torch.inverse(decomposed))
        logdet = decomposed:diag():log():sum()
    end
    transformed:apply(function(a) return randomkit.gaussianLogPDF(a, 0, 1) end)
    local result = transformed:sum(2) - logdet -- by independence
    if scalarResult then
        return result[1][1]
    else
        return result
    end
end

function randomkit.multivariateGaussianPDF(...)
    local r = randomkit.multivariateGaussianLogPDF(...)
    if type(r) == 'number' then
        return math.exp(r)
    else
        return r:exp()
    end
end

function randomkit.multivariateGaussianRand(...)
    error("Not implemented")
end
