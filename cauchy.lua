function randomkit.cauchyPDF(x, a, b)
    return 1.0 / math.pi * b / (math.pow(x - a, 2) + b*b)
end

function randomkit.cauchyLogPDF(x, a, b)
    return -math.log(math.pi) + math.log(b) - math.log(math.pow(x - a, 2) + b*b)
end

function randomkit.cauchyCDF(x, a, b)
    return 1.0 / math.pi * cephes.atan((x - a) / b) + 0.5
end

function randomkit.cauchyQuantile(p, a, b)
    error("Not implemented")
end

function randomkit.cauchyRand(...)
    error("Not implemented")
end
