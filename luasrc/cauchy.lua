function distributions.cauchyPDF(x, a, b)
    return 1.0 / math.pi * b / (math.pow(x - a, 2) + b*b)
end

function distributions.cauchyLogPDF(x, a, b)
    return -math.log(math.pi) + math.log(b) - math.log(math.pow(x - a, 2) + b*b)
end

function distributions.cauchyCDF(x, a, b)
    return 1.0 / math.pi * cephes.atan((x - a) / b) + 0.5
end

function distributions.cauchyQuantile(p, a, b)
    error("Not implemented")
end

function distributions.cauchyRand(...)
    error("Not implemented")
end
