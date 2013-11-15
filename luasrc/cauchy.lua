distributions.cauchy = {}

function distributions.cauchy.pdf(x, a, b)
    return 1.0 / math.pi * b / (math.pow(x - a, 2) + b*b)
end

function distributions.cauchy.logpdf(x, a, b)
    return -math.log(math.pi) + math.log(b) - math.log(math.pow(x - a, 2) + b*b)
end

function distributions.cauchy.cdf(x, a, b)
    return 1.0 / math.pi * cephes.atan((x - a) / b) + 0.5
end

function distributions.cauchy.qtl(p, a, b)
    error("Not implemented")
end

function distributions.cauchy.rnd(...)
    error("Not implemented")
end
