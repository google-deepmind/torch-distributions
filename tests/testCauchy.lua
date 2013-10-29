require 'randomkit'
local myTest = {}
local tester = torch.Tester()

function myTest.cauchyPDF()
    tester:assertalmosteq(randomkit.cauchyPDF(-0.5, 0, 1), 0.254648, 1e-5)
    tester:assertalmosteq(randomkit.cauchyPDF(0,    0, 1), 1.0/math.pi, 1e-5)
    tester:assertalmosteq(randomkit.cauchyPDF(1,    0, 1), 0.5/math.pi, 1e-5)
    tester:assertalmosteq(randomkit.cauchyPDF(4,    0, 1), 0.0187241, 1e-5)

    tester:assertalmosteq(randomkit.cauchyPDF(-0.5, 2, 3), 0.062618, 1e-5)
    tester:assertalmosteq(randomkit.cauchyPDF(1,    2, 3), 0.095493, 1e-5)
    tester:assertalmosteq(randomkit.cauchyPDF(2,    2, 3), 0.106103, 1e-5)
    tester:assertalmosteq(randomkit.cauchyPDF(4,    2, 3), 0.0734561, 1e-5)
end

function myTest.cauchyCDF()
    tester:assertalmosteq(randomkit.cauchyCDF(-0.5, 0, 1), 0.352416, 1e-5)
    tester:assertalmosteq(randomkit.cauchyCDF(0,    0, 1), 0.5, 1e-5)
    tester:assertalmosteq(randomkit.cauchyCDF(1,    0, 1), 0.75, 1e-5)
    tester:assertalmosteq(randomkit.cauchyCDF(4,    0, 1), 0.922021, 1e-5)

    tester:assertalmosteq(randomkit.cauchyCDF(-0.5, 2, 3), 0.278858, 1e-5)
    tester:assertalmosteq(randomkit.cauchyCDF(1,    2, 3), 0.397584, 1e-5)
    tester:assertalmosteq(randomkit.cauchyCDF(2,    2, 3), 0.5, 1e-5)
    tester:assertalmosteq(randomkit.cauchyCDF(4,    2, 3), 0.687167, 1e-5)
end

tester:add(myTest)
tester:run()
