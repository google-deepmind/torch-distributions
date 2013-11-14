require 'distributions'
local myTest = {}
local tester = torch.Tester()

function myTest.cauchyPDF()
    tester:assertalmosteq(distributions.cauchyPDF(-0.5, 0, 1), 0.254648, 1e-5)
    tester:assertalmosteq(distributions.cauchyPDF(0,    0, 1), 1.0/math.pi, 1e-5)
    tester:assertalmosteq(distributions.cauchyPDF(1,    0, 1), 0.5/math.pi, 1e-5)
    tester:assertalmosteq(distributions.cauchyPDF(4,    0, 1), 0.0187241, 1e-5)

    tester:assertalmosteq(distributions.cauchyPDF(-0.5, 2, 3), 0.062618, 1e-5)
    tester:assertalmosteq(distributions.cauchyPDF(1,    2, 3), 0.095493, 1e-5)
    tester:assertalmosteq(distributions.cauchyPDF(2,    2, 3), 0.106103, 1e-5)
    tester:assertalmosteq(distributions.cauchyPDF(4,    2, 3), 0.0734561, 1e-5)
end

function myTest.cauchyLogPDF()
    tester:assertalmosteq(distributions.cauchyLogPDF(-0.5, 0, 1), math.log(0.254648), 1e-5)
    tester:assertalmosteq(distributions.cauchyLogPDF(0,    0, 1), math.log(1.0/math.pi), 1e-5)
    tester:assertalmosteq(distributions.cauchyLogPDF(1,    0, 1), math.log(0.5/math.pi), 1e-5)
    tester:assertalmosteq(distributions.cauchyLogPDF(4,    0, 1), math.log(0.0187241), 1e-5)
    tester:assertalmosteq(distributions.cauchyLogPDF(-0.5, 2, 3), math.log(0.062618), 1e-5)
    tester:assertalmosteq(distributions.cauchyLogPDF(1,    2, 3), math.log(0.095493), 1e-5)
    tester:assertalmosteq(distributions.cauchyLogPDF(2,    2, 3), math.log(0.106103), 1e-5)
    tester:assertalmosteq(distributions.cauchyLogPDF(4,    2, 3), math.log(0.0734561), 1e-5)
end

function myTest.cauchyCDF()
    tester:assertalmosteq(distributions.cauchyCDF(-0.5, 0, 1), 0.352416, 1e-5)
    tester:assertalmosteq(distributions.cauchyCDF(0,    0, 1), 0.5, 1e-5)
    tester:assertalmosteq(distributions.cauchyCDF(1,    0, 1), 0.75, 1e-5)
    tester:assertalmosteq(distributions.cauchyCDF(4,    0, 1), 0.922021, 1e-5)

    tester:assertalmosteq(distributions.cauchyCDF(-0.5, 2, 3), 0.278858, 1e-5)
    tester:assertalmosteq(distributions.cauchyCDF(1,    2, 3), 0.397584, 1e-5)
    tester:assertalmosteq(distributions.cauchyCDF(2,    2, 3), 0.5, 1e-5)
    tester:assertalmosteq(distributions.cauchyCDF(4,    2, 3), 0.687167, 1e-5)
end

tester:add(myTest)
tester:run()
