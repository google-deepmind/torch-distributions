require 'randomkit'
local seedTest = {}
local tester = torch.Tester()

function seedTest.manualSeed()

end

function seedTest.manualSeed()
    local seed = 1234567890

    local function test_generator(f, s)
        s = s or ''
        torch.manualSeed(seed)
        local before = f()
        torch.manualSeed(seed)
        local after = f()
        tester:assertTensorEq(before, after, 1e-16, 'manualSeed not generating same sequence for generator ' .. s)
    end

   -- Try on uniform
   -- Make sure we do not ruin torch's generators
   test_generator(function() return torch.rand(10) end, 'torch.rand')
   test_generator(function() return randomkit.double(torch.Tensor(10)) end, 'randomkit.double')

   -- Try on Gauss with odd number of variates, trickiest case due to dangling gaussian
   local oddN = 3
   test_generator(function() return randomkit.gauss(torch.Tensor(oddN)) end, 'randomkit.gauss')
   test_generator(function() return torch.randn(oddN) end, 'torch.randn')

end

function seedTest.RNGState()
   local ignored, state, stateCloned, before, after

    local function test_generator(f, s)
        s = s or ''
        ignored = f()
        state = torch.getRNGState()
        stateCloned = state:clone()
        before = f()
        tester:assert(state:ne(stateCloned):long():sum() == 0, 'RNG (supposedly cloned) state has changed after random number generation')
        torch.setRNGState(state)
        after = f()
        tester:assertTensorEq(before, after, 1e-16, 'getRNGState/setRNGState not generating same sequence for generator ' .. s)
    end

   -- Try on uniform
   -- Make sure we do not ruin torch's generators
   test_generator(function() return torch.rand(10) end, 'torch.rand')
   test_generator(function() return randomkit.double(torch.Tensor(10)) end, 'randomkit.double')

   -- Try on Gauss with odd number of variates, trickiest case due to dangling gaussian
   local oddN = 3
   test_generator(function() return randomkit.gauss(torch.Tensor(oddN)) end, 'randomkit.gauss')
   test_generator(function() return torch.randn(oddN) end, 'torch.randn')
end

function seedTest.manualSeedGauss()
   -- Test for issue #191
   local before, after

   torch.manualSeed(1234567890)
   before = torch.randn(1)

   torch.manualSeed(1234567890)
   after = torch.randn(1)

   tester:assertTensorEq(before, after, 1e-16, 'manualSeed not generating same odd gaussian sequence')
end




tester:add(seedTest)
tester:run()
