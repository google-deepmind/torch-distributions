require 'randomkit'
local ffi = require 'ffi'

local myTest = {}
local tester = torch.Tester()

function myTest.testFFICall()
    local state = ffi.new('rk_state')
    randomkit.ffi.rk_seed(0, state)
    local x
    for i=1, 10 do
        tester:assert(tonumber(randomkit.ffi.rk_binomial(state, 10, 0.4)))
    end
end

function myTest.testWrappedCall()
    local N = 1000
    local x = torch.Tensor(N)
    local y = torch.Tensor(N)
    local state = torch.getRNGState()
    for i=1,N do
        x[i] = randomkit.binomial(10, 0.4)
    end
    torch.setRNGState(state)
    for i=1,N do
        y[i] = randomkit.binomial(10, 0.4)
    end
    tester:assertTensorEq(x,y,1e-16,'RK sequence is not a deterministic function of state')
    -- TODO: check distribtution

end

tester:add(myTest)
tester:run()
