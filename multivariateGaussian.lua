function randomkit.multivariateGaussianPDF(x, mu, sigma)
    -- TODO multiple samples (vectorised)
    mu:resize(1, mu:nElement())
    local input = x:resize(1, x:nElement())
    local decomposed = torch.potrf(sigma):triu() -- TODO remove triu as torch will be fixed
    local inverse = torch.inverse(decomposed)
    local det = decomposed:diag():prod(1)[1]
    local transformed = torch.mm(input:add(mu*-1), inverse)
    transformed:apply(function(a) return randomkit.gaussianPDF(a, 0, 1) end)
    outputs = transformed
    return outputs:prod(2)[1][1] / det -- by independence
end

function randomkit.multivariateGaussianLogPDF(x, mu, sigma)
    error("Not implemented")
end

function randomkit.multivariateGaussianRand(...)
    error("Not implemented")
end
