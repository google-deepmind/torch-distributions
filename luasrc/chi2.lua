function randomkit.chi2PDF(x, dof)
    return math.pow(x, dof/2 - 1) * cephes.exp(-x/2) / (math.pow(2, dof/2) * cephes.gamma(dof/2))
end

function randomkit.chi2LogPDF(x, dof)
    return -dof/2*cephes.log(2) - cephes.lgam(dof/2) + (dof/2 - 1)*cephes.log(x) - x/2
end

function randomkit.chi2CDF(x, dof)
    return cephes.chdtr(dof, x)
end

function randomkit.chi2Quantile(p, dof)
    error("Not implemented")
end

function randomkit.chi2Rand(...)
    error("Not implemented")
end
