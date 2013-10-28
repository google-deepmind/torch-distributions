--[[ This file is for functions that are not direct wraps of the randomkit C routines ]]

function randomkit.bytes(...)
    local result, params = randomkit._check1DParams(0, torch.ByteTensor, ...)
    if torch.typename(result) ~= "torch.ByteTensor" then
        error("randomkit.bytes() can only store into a ByteTensor!")
    end
    local dataPtr = torch.data(result)
    randomkit.ffi.rk_fill(dataPtr, result:nElement(), state)
    return result
end

