distributions.chi2 = {}

function distributions.chi2.pdf(x, dof)
    return math.pow(x, dof/2 - 1) * cephes.exp(-x/2) / (math.pow(2, dof/2) * cephes.gamma(dof/2))
end

function distributions.chi2.logpdf(x, dof)
    return -dof/2*cephes.log(2) - cephes.lgam(dof/2) + (dof/2 - 1)*cephes.log(x) - x/2
end

function distributions.chi2.cdf(x, dof)
    return cephes.chdtr(dof, x)
end

function distributions.chi2.qtl(p, dof)
    error("Not implemented")
end

function distributions.chi2.rnd(...)
    error("Not implemented")
end
