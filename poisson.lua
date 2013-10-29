require 'cephes'

function randomkit.poissonPDF(x, lambda)
    return cephes.pow(lambda, x) / cephes.fac(x) * cephes.exp(-lambda)
end

function randomkit.poissonLogPDF(x, lambda)
    error("Not implemented")
end

function randomkit.poissonCDF(x, lambda)
    error("Not implemented")
end

function randomkit.poissonQuantile(p, lambda)
    error("Not implemented")
end
