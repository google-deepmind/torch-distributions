require 'totem'
require 'distributions'
require 'pl.strict'

local myTests = {}
local tester = totem.Tester()
torch.manualSeed(1234567890)

function myTests.testPDF()
end

function myTests.testRND()
end

tester:add(myTests)
return tester:run()