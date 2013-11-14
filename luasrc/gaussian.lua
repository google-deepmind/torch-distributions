function distributions.gaussianPDF(x, mu, sigma)
    return cephes.exp(-.5 * (x-mu)*(x-mu)/(sigma*sigma)) / math.sqrt(2.0*math.pi*sigma*sigma)
end

function distributions.gaussianLogPDF(x, mu, sigma)
    return -.5 * (x-mu)*(x-mu)/(sigma*sigma) - 0.5 * math.log(2*math.pi*sigma*sigma)
end

function distributions.gaussianCDF(x, mu, sigma)
    return 0.5 * (1.0 + cephes.erf((x-mu)/math.sqrt(2*sigma*sigma)))
end

function distributions.gaussianQuantile(p, mu, sigma)
    error("Not implemented")
end

function distributions.gaussianRand(...)
    error("Not implemented")
end
