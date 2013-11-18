distributions.mvn = {}

function distributions.mvn.logpdf(x, mu, sigma, options)
    options = options or {}
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
    local decomposed

    -- For a diagonal covariance matrix, we allow passing a vector of the diagonal entries
    if sigma:dim() == 1 then
        local D = sigma:size(1)
        decomposed  = sigma
        if not options.cholesky then
            decomposed:sqrt()
        end
        logdet = decomposed:clone():log():sum()
        transformed = torch.cdiv(x, decomposed:resize(1, D):expand(nResults, D))
    else
        if not options.cholesky then
            decomposed = torch.potrf(sigma):triu() -- TODO remove triu as torch will be fixed
        else
            decomposed = sigma
        end
        transformed = torch.gesv(x:t(), decomposed:t()):t()
        logdet = decomposed:diag():log():sum()
    end
    transformed:apply(function(a) return distributions.norm.logpdf(a, 0, 1) end)
    local result = transformed:sum(2) - logdet -- by independence
    if scalarResult then
        return result[1][1]
    else
        return result
    end
end

function distributions.mvn.pdf(...)
    local r = distributions.mvn.logpdf(...)
    if type(r) == 'number' then
        return math.exp(r)
    else
        return r:exp()
    end
end

function distributions.mvn.rnd(...)
    local nArgs = select("#", ...)
    local resultTensor

    local n -- number of samples
    local d -- number of dimensions for the Gaussian
    local mu -- mean
    local sigma -- covariance matrix

    local function inferDimension(sigma)
        if sigma:dim() == 1 then
            -- diagonal, and only one covariance matrix
            return sigma:size(1)
        else
            return sigma:size(2)
        end
        return d
    end

    local options = {}
    -- Is the last argument an options table?
    if type(select(nArgs, ...)) == 'table' then
        options = select(nArgs, ...)
        nArgs = nArgs - 1
    end

    if nArgs == 2 then -- mu, sigma only: return one sample
        n = 1
        mu = torch.Tensor(select(1, ...))
        sigma = torch.Tensor(select(2, ...))
        d = inferDimension(sigma)
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
                error("Incoherent parameter sizes for mvn.rnd")
            end
        end
        if not nParams and sigma:dim() == 3 then
            nParams = sigma:size(1)
        end
        d = inferDimension(sigma)
        if type(resultInfo) == 'number' then
            n = resultInfo
            resultTensor = torch.Tensor(n, d)
            if nParams and nParams ~= n then
                error("Parameter sizes do not match number of samples requested")
            end
        elseif distributions._isTensor(resultInfo) then
            resultTensor = resultInfo
            if nParams then
                n = nParams
            else
                n = resultTensor:nElement() / d
            end
            resultTensor:resize(n, d)
        else
            error("Unable to understand first argument for mvn.rnd - should be an integer number of samples to be returned, or a result tensor")
        end

    else
        error("Invalid arguments for mvn.rnd().\
        Expecting [N|ResultTensor,] mu, sigma [, options].")
    end

    -- Now make our inputs all tensors, for simplicity
    if mu:dim() == 1 then
        mu:resize(1, mu:nElement())
    end
    if sigma:dim() == 1 then
        if mu:size(2) ~= sigma:size(1) then
            error("mvn.rnd: inconsistent sizes for mu and sigma")
        end
        sigma:resize(1, d)
    elseif sigma:dim() == 2 then
        -- either DxD or NxD
        if sigma:size(1) == sigma:size(2) then
            if n == d then
                error("mvn.rnd: ambiguous covariance input")
            end
        end

        if mu:size(2) ~= sigma:size(1) or mu:size(2) ~= sigma:size(2) then
            error("mvn.rnd: inconsistent sizes for mu and sigma")
        end
        sigma:resize(1, d, d)
    elseif sigma:dim() == 3 then
        if mu:size(2) ~= d or sigma:size(2) ~= d or sigma:size(3) ~= d then
            error("mvn.rnd: inconsistent sizes for mu and sigma")
        end
    end
    if mu:size(1) == 1 then
        mu = mu:expand(n, d)
    end

    local function sampleFromDistribution(resultTensor, x, mu, sigma)
        local resultSize = resultTensor:size()
        local y
        if sigma:dim() == 2 then
            -- TODO: when Lapack's pstrf will be wrapped in Torch,
            -- use that instead of Cholesky with SVD failsafe
            if options.cholesky then
                y = torch.mm(x, sigma)
            else
                local fullRank, decomposed = pcall(function() return torch.potrf(sigma):triu() end)
                if fullRank then
                    -- Definite positive matrix: use Cholesky
                    y = torch.mm(x, decomposed)
                else
                    -- Rank-deficient matrix: fall back on SVD
                    local u, s, v = torch.svd(sigma)
                    local tmp = torch.cmul(x, s:sqrt():resize(1, d):expand(n, d))
                    y = torch.mm(tmp, v)
                end
            end

        else
            -- diagonal sigma
            local decomposed
            decomposed = sigma:clone()
            if not options.cholesky then
                decomposed:sqrt()
            end
            y = torch.cmul(decomposed:resize(1,d):expand(n,d), x)
        end

        torch.add(resultTensor, y, mu):resize(resultSize)

    end

    local x = torch.Tensor(n,d)
    randomkit.gauss(x)
    if sigma:size(1) == 1 then
        sampleFromDistribution(resultTensor, x, mu, sigma[1])
        return resultTensor
    else
        for k = 1, n do
            sampleFromDistribution(resultTensor[k], x[k]:resize(1, d), mu[k], sigma[k])
        end
        return resultTensor
    end
end
