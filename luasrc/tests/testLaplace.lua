require 'distributions'
local myTest = {}
local tester = torch.Tester()

function myTest.laplacePDF()
    tester:assertalmosteq(distributions.laplacePDF(-2, 0, 1), 0.06766764161830634, 1e-15)
    tester:assertalmosteq(distributions.laplacePDF(0,  0, 1), 0.5, 1e-15)
    tester:assertalmosteq(distributions.laplacePDF(1,  0, 1), 0.18393972058572116, 1e-15)
    tester:assertalmosteq(distributions.laplacePDF(6,  0, 1), 0.00123937608833317, 1e-15)
    tester:assertalmosteq(distributions.laplacePDF(-2, 0.5, 4), 0.0669077, 1e-5)
    tester:assertalmosteq(distributions.laplacePDF(0,  0.5, 4), 0.110312, 1e-5)
    tester:assertalmosteq(distributions.laplacePDF(1,  0.5, 4), 0.110312, 1e-5)
    tester:assertalmosteq(distributions.laplacePDF(6,  0.5, 4), 0.0316049, 1e-5)
end

function myTest.laplaceLogPDF()
    tester:assertalmosteq(distributions.laplaceLogPDF(-2, 0, 1),   math.log(0.06766764161830634), 1e-14)
    tester:assertalmosteq(distributions.laplaceLogPDF(0,  0, 1),   math.log(0.5), 1e-14)
    tester:assertalmosteq(distributions.laplaceLogPDF(1,  0, 1),   math.log(0.18393972058572116), 1e-14)
    tester:assertalmosteq(distributions.laplaceLogPDF(6,  0, 1),   math.log(0.00123937608833317), 1e-14)
    tester:assertalmosteq(distributions.laplaceLogPDF(-2, 0.5, 4), math.log(0.0669077), 1e-5)
    tester:assertalmosteq(distributions.laplaceLogPDF(0,  0.5, 4), math.log(0.110312), 1e-5)
    tester:assertalmosteq(distributions.laplaceLogPDF(1,  0.5, 4), math.log(0.110312), 1e-5)
    tester:assertalmosteq(distributions.laplaceLogPDF(6,  0.5, 4), math.log(0.0316049), 1e-5)
end

function myTest.laplaceCDF()
    tester:assertalmosteq(distributions.laplaceCDF(-2, 0, 1), 0.0676676416183063, 1e-15)
    tester:assertalmosteq(distributions.laplaceCDF(0,  0, 1), 0.5, 1e-15)
    tester:assertalmosteq(distributions.laplaceCDF(1,  0, 1), 0.8160602794142788, 1e-15)
    tester:assertalmosteq(distributions.laplaceCDF(6,  0, 1), 0.9987606239116668, 1e-15)
    tester:assertalmosteq(distributions.laplaceCDF(-2, 0.5, 4), 0.267631, 1e-5)
    tester:assertalmosteq(distributions.laplaceCDF(0,  0.5, 4), 0.441248, 1e-5)
    tester:assertalmosteq(distributions.laplaceCDF(1,  0.5, 4), 0.558752, 1e-5)
    tester:assertalmosteq(distributions.laplaceCDF(6,  0.5, 4), 0.87358, 1e-5)
end

tester:add(myTest)
tester:run()
