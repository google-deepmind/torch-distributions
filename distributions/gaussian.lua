distributions.norm = {}

function distributions.norm.pdf(x, mu, sigma)
    return cephes.exp(-.5 * (x-mu)*(x-mu)/(sigma*sigma)) / math.sqrt(2.0*math.pi*sigma*sigma)
end

function distributions.norm.logpdf(x, mu, sigma)
    return -.5 * (x-mu)*(x-mu)/(sigma*sigma) - 0.5 * math.log(2*math.pi*sigma*sigma)
end

function distributions.norm.cdf(x, mu, sigma)
    return 0.5 * (1.0 + cephes.erf((x-mu)/math.sqrt(2*sigma*sigma)))
end

function distributions.norm.qtl(p, mu, sigma)
    return cephes.ndtri(p) * sigma + mu
end

function distributions.norm.rnd(...)
    error("Not implemented")
end
