require 'distributions'
local myTest = {}
local tester = torch.Tester()

function myTest.poissonPDF()
    tester:assertalmosteq(distributions.poissonPDF(0, 2), 0.135335283236612, 1e-15)
    tester:assertalmosteq(distributions.poissonPDF(0, 1), 0.367879441171442, 1e-15)
    tester:assertalmosteq(distributions.poissonPDF(1, 1), 0.367879441171442, 1e-15)
    tester:assertalmosteq(distributions.poissonPDF(2, 1), 0.183939720585721, 1e-15)
    tester:assertalmosteq(distributions.poissonPDF(0, 10), 0.000045399929762, 1e-15)
    tester:assertalmosteq(distributions.poissonPDF(1, 10), 0.000453999297625, 1e-15)
    tester:assertalmosteq(distributions.poissonPDF(2, 10), 0.002269996488124, 1e-15)
end

function myTest.poissonPDFVectorized()
    local xs = torch.Tensor({0, 0, 1, 2, 0, 1, 2})
    local lambdas = torch.Tensor({2, 1, 1, 1, 10, 10, 10})
    local expected = torch.Tensor({0.135335283236612, 0.367879441171442, 0.367879441171442, 0.183939720585721, 0.000045399929762, 0.000453999297625, 0.002269996488124})
    local result = distributions.poissonPDF(xs, lambdas)
    tester:assert(result and torch.typename(result) == 'torch.DoubleTensor')
    tester:assertTensorEq(result, expected, 1e-15, "poisson pdf results should match expected, for call with vectors of samples and parameters")
end

function myTest.poissonPDFVectorized2()
    local xs = torch.Tensor({0, 1, 2})
    local lambdas = torch.Tensor({1})
    local expected = torch.Tensor({0.367879441171442, 0.367879441171442, 0.183939720585721})
    local result = distributions.poissonPDF(xs, lambdas)
    tester:assert(result and torch.typename(result) == 'torch.DoubleTensor')
    tester:assertTensorEq(result, expected, 1e-15, "poisson pdf results should match expected, for call with vector of samples")
end

function myTest.poissonPDFVectorized3()
    local xs = torch.Tensor({0})
    local lambdas = torch.Tensor({1, 2, 10})
    local expected = torch.Tensor({0.367879441171442, 0.135335283236612, 0.000045399929762})
    local result = distributions.poissonPDF(xs, lambdas)
    tester:assert(result and torch.typename(result) == 'torch.DoubleTensor')
    tester:assertTensorEq(result, expected, 1e-15, "poisson pdf results should match expected, for call with vector of parameters")
end

function myTest.poissonLogPDF()
    tester:assertalmosteq(distributions.poissonLogPDF(0, 2), -2, 1e-13)
    tester:assertalmosteq(distributions.poissonLogPDF(0, 1), -1, 1e-13)
    tester:assertalmosteq(distributions.poissonLogPDF(1, 1), -1, 1e-13)
    tester:assertalmosteq(distributions.poissonLogPDF(2, 1), -1.6931471805599, 1e-13)
    tester:assertalmosteq(distributions.poissonLogPDF(0, 10), -10, 1e-13)
    tester:assertalmosteq(distributions.poissonLogPDF(1, 10), -7.69741490700595, 1e-13)
    tester:assertalmosteq(distributions.poissonLogPDF(2, 10), -6.08797699457185, 1e-13)
end

function myTest.poissonCDF()
    tester:assertalmosteq(distributions.poissonCDF(0.1, 2), 0.135335283236612, 1e-15)
    tester:assertalmosteq(distributions.poissonCDF(0, 1), 0.367879441171442, 1e-15)
    tester:assertalmosteq(distributions.poissonCDF(1, 1), 0.735758882342884, 1e-15)
    tester:assertalmosteq(distributions.poissonCDF(2, 1), 0.919698602928605, 1e-15)
    tester:assertalmosteq(distributions.poissonCDF(0, 10), 0.000045399929762, 1e-15)
    tester:assertalmosteq(distributions.poissonCDF(1, 10), 0.000499399227387, 1e-15)
    tester:assertalmosteq(distributions.poissonCDF(2, 10), 0.002769395715512, 1e-15)
end

tester:add(myTest)
tester:run()
