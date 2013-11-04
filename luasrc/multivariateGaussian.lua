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
    local nArgs = select("#", ...)
    local resultTensor

    local n -- number of samples
    local d -- number of dimensions for the Gaussian
    local mu -- mean
    local sigma -- covariance matrix

    if nArgs == 2 then -- mu, sigma only: return one sample
        n = 1
        mu = torch.Tensor(select(1, ...))
        sigma = torch.Tensor(select(2, ...))
        d = sigma:size(2)
        resultTensor = torch.Tensor(d)
    elseif nArgs == 3 then -- RESULT, mu, sigma - where result is either a number or an output tensor
        local resultInfo = select(1, ...)
        mu = torch.Tensor(select(2, ...))
        sigma = torch.Tensor(select(3, ...))
        -- If we have non-constant parameters, get the number of results to return from there
        local nParams
        if mu:dim() ~= 1 then
            nParams = mu:size(1)
            if sigma:dim() == 3 and sigma:size(1) ~= nParams then
                error("Incoherent parameter sizes for multivariateGaussianRand")
            end
        end
        if not nParams and sigma:dim() == 3 then
            nParams = sigma:size(1)
        end
        if type(resultInfo) == 'number' then
            n = resultInfo
            d = sigma:size(1)
            resultTensor = torch.Tensor(n, d)
            if nParams and nParams ~= n then
                error("Parameter sizes do not match number of samples requested")
            end
        elseif randomkit._isTensor(resultInfo) then
            resultTensor = resultInfo
            d = sigma:size(1)
            if nParams then
                n = nParams
            else
                n = resultTensor:nElements() / d
            end
        else
            error("Unable to understand first argument for multivariateGaussianRand - should be an integer number of samples to be returned, or a result tensor")
        end

    else
        error("Invalid arguments for multivariateGaussianRand().\
        Should be (mu, sigma), or (N, mu, sigma), or (ResultTensor, mu, sigma).")
    end

    -- Now make our inputs all tensors, for simplicity
    if mu:dim() == 1 then
        mu:resize(1, mu:nElement())
    end
    if sigma:dim() == 2 then
        sigma:resize(1, d, d)
    end
    if mu:size(2) ~= sigma:size(2) then
        error("multivariateGaussianRand: inconsistent sizes for mu and sigma")
    end
    if mu:size(1) == 1 then
        mu = mu:expand(n, d)
    end
    if sigma:size(1) == 1 then

        local decomposed = torch.potrf(sigma[1]):triu() -- TODO remove triu as torch will be fixed
        local s = torch.mm(torch.randn(n, d), decomposed) + mu
        resultTensor:copy(s)

        return resultTensor

    else
        error("Multiple covariance matrices: not implemented")
        --[[ TODO multiple sigmas
        for k = 1, n do
            local decomposed = torch.potrf(sigma[k]):triu() -- TODO remove triu as torch will be fixed
            local r = torch.mm(torch.randn(n, d), decomposed) + mu
        end
        --]]

    end
end

