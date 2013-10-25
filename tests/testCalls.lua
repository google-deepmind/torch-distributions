require 'randomkit'
local ffi = require 'ffi'

local myTest = {}
local tester = torch.Tester()

function myTest.test()
    local state = ffi.new('rk_state')
    randomkit.rk_seed(190983, state)
    local x
    for i=1, 10 do
        x = randomkit.rk_binomial(state, 10, 0.4)
        print('type of x:', ffi.typeof(x))
        print('x:', tonumber(x))
    end
end

tester:add(myTest)
tester:run()
