distributions.laplace = {}

function distributions.laplace.pdf(x, loc, scale)
    return 1.0/(2.0*scale)*cephes.exp(-math.abs(x - loc)/scale)
end

function distributions.laplace.logpdf(x, loc, scale)
    return -math.log(2.0*scale) - math.abs(x - loc) / scale
end

function distributions.laplace.cdf(x, loc, scale)
    if x < loc then
        return 0.5 * math.exp((x - loc) / scale)
    else
        return 1 - 0.5 * math.exp((loc - x) / scale)
    end
end

function distributions.laplace.qtl(p, loc, scale)
    error("Not implemented")
end

function distributions.laplace.rnd(...)
    error("Not implemented")
end
