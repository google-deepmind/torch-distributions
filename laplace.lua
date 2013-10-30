function randomkit.laplacePDF(x, loc, scale)
    return 1.0/(2.0*scale)*cephes.exp(-math.abs(x - loc)/scale)
end

function randomkit.laplaceLogPDF(x, loc, scale)
    return -math.log(2.0*scale) - math.abs(x - loc) / scale
end

function randomkit.laplaceCDF(x, loc, scale)
    if x < loc then
        return 0.5 * math.exp((x - loc) / scale)
    else
        return 1 - 0.5 * math.exp((loc - x) / scale)
    end
end

function randomkit.laplaceQuantile(p, loc, scale)
    error("Not implemented")
end

function randomkit.laplaceRand(...)
    error("Not implemented")
end
