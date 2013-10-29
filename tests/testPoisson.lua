require 'randomkit'
local myTest = {}
local tester = torch.Tester()

function myTest.poissonPDF()
    tester:assertalmosteq(randomkit.poissonPDF(0, 2), 0.135335283236612, 1e-15)
    tester:assertalmosteq(randomkit.poissonPDF(0, 1), 0.367879441171442, 1e-15)
    tester:assertalmosteq(randomkit.poissonPDF(1, 1), 0.367879441171442, 1e-15)
    tester:assertalmosteq(randomkit.poissonPDF(2, 1), 0.183939720585721, 1e-15)
    tester:assertalmosteq(randomkit.poissonPDF(0, 10), 0.000045399929762, 1e-15)
    tester:assertalmosteq(randomkit.poissonPDF(1, 10), 0.000453999297625, 1e-15)
    tester:assertalmosteq(randomkit.poissonPDF(2, 10), 0.002269996488124, 1e-15)
end

tester:add(myTest)
tester:run()
