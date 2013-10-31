function randomkit.multivariateGaussianPDF(x, mu, sigma)
    x = torch.Tensor(x)
    mu = torch.Tensor(mu)

    local scalarResult = (x:dim() == 1) and (mu:dim() == 1)

    if x:dim() == 1 then
        x:resize(1, x:nElement())
    end
    if mu:dim() == 1 then
        mu:resize(1, mu:nElement())
    end
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

    local decomposed = torch.potrf(sigma):triu() -- TODO remove triu as torch will be fixed
    local inverse = torch.inverse(decomposed)
    local det = decomposed:diag():prod(1)[1]
    local transformed = torch.mm(x:add(-1, mu), inverse)
    transformed:apply(function(a) return randomkit.gaussianPDF(a, 0, 1) end)
    local result = transformed:prod(2) / det -- by independence

    if scalarResult then
        return result[1][1]
    else
        return result
    end
end

function randomkit.multivariateGaussianLogPDF(x, mu, sigma)
    error("Not implemented")
end

function randomkit.multivariateGaussianRand(...)
    error("Not implemented")
end
