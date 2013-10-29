require 'randomkit'
local myTest = {}
local tester = torch.Tester()

function myTest.poissonPDF()
    tester:assertalmosteq(randomkit.poissonPDF(0, 2), 0.1353352832366126918939994949, 1e-15)
    tester:assertalmosteq(randomkit.poissonPDF(0, 1), 0.3678794411714423215955237701, 1e-15)
    tester:assertalmosteq(randomkit.poissonPDF(1, 1), 0.3678794411714423215955237701, 1e-15)
    tester:assertalmosteq(randomkit.poissonPDF(2, 1), 0.1839397205857211607977618850, 1e-15)
end

tester:add(myTest)
tester:run()
