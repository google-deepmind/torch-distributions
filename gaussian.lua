function randomkit.gaussianPDF(x, mu, sigma)
    local normalizingFactor = 1.0 / math.sqrt(2.0*math.pi*sigma*sigma)
    return cephes.exp(-(x-mu)*(x-mu)/(2*sigma*sigma)) * normalizingFactor
end

function randomkit.gaussianLogPDF(x, mu, sigma)
    error("Not implemented")
end

function randomkit.gaussianCDF(x, mu, sigma)
    return 0.5 * (1.0 + cephes.erf((x-mu)/math.sqrt(2*sigma*sigma)))
end

function randomkit.gaussianQuantile(p, mu, sigma)
    error("Not implemented")
end

function randomkit.gaussianRand(...)
    error("Not implemented")
end
