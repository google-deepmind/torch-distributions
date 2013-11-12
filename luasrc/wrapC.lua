local ffi = require("ffi")
require 'torchffi'

randomkit.ffi = ffi.load(package.searchpath('librandomkit', package.cpath))

ffi.cdef[[
typedef struct rk_state_
{
    unsigned long key[624];
    int pos;
    int has_gauss; /* !=0: gauss contains a gaussian deviate */
    double gauss;

    /* The rk_state structure has been extended to store the following
     * information for the binomial generator. If the input values of n or p
     * are different than nsave and psave, then the other parameters will be
     * recomputed. RTK 2005-09-02 */

    int has_binomial; /* !=0: following parameters initialized for
                              binomial */
    double psave;
    long nsave;
    double r;
    double q;
    double fm;
    long m;
    double p1;
    double xm;
    double xl;
    double xr;
    double c;
    double laml;
    double lamr;
    double p2;
    double p3;
    double p4;

}
rk_state;

typedef enum {
    RK_NOERR = 0, /* no error */
    RK_ENODEV = 1, /* no RK_DEV_RANDOM device */
    RK_ERR_MAX = 2
} rk_error;

/* error strings */
 char *rk_strerror[2];
]]

-- Declare the functions, spearately since this string
-- is reused for automatic wrapping
fun_cdef = [[
 void rk_seed(unsigned long seed, rk_state *state);
 long rk_random(rk_state *state);
 long rk_long(rk_state *state);
 unsigned long rk_ulong(rk_state *state);
 unsigned long rk_interval(unsigned long max, rk_state *state);
 double rk_double(rk_state *state);
 void rk_fill(void *buffer, size_t size, rk_state *state);
 rk_error rk_devfill(void *buffer, size_t size, int strong);
 rk_error rk_altfill(void *buffer, size_t size, int strong, rk_state *state);
 double rk_gauss(rk_state *state);
 double rk_normal(rk_state *state, double loc, double scale);
 double rk_standard_exponential(rk_state *state);
 double rk_exponential(rk_state *state, double scale);
 double rk_uniform(rk_state *state, double loc, double scale);
 double rk_standard_gamma(rk_state *state, double shape);
 double rk_gamma(rk_state *state, double shape, double scale);
 double rk_beta(rk_state *state, double a, double b);
 double rk_chisquare(rk_state *state, double df);
 double rk_noncentral_chisquare(rk_state *state, double df, double nonc);
 double rk_f(rk_state *state, double dfnum, double dfden);
 double rk_noncentral_f(rk_state *state, double dfnum, double dfden, double nonc);
 long rk_binomial(rk_state *state, long n, double p);
 long rk_binomial_btpe(rk_state *state, long n, double p);
 long rk_binomial_inversion(rk_state *state, long n, double p);
 long rk_negative_binomial(rk_state *state, double n, double p);
 long rk_poisson(rk_state *state, double lam);
 long rk_poisson_mult(rk_state *state, double lam);
 long rk_poisson_ptrs(rk_state *state, double lam);
 double rk_standard_cauchy(rk_state *state);
 double rk_standard_t(rk_state *state, double df);
 double rk_vonmises(rk_state *state, double mu, double kappa);
 double rk_pareto(rk_state *state, double a);
 double rk_weibull(rk_state *state, double a);
 double rk_power(rk_state *state, double a);
 double rk_laplace(rk_state *state, double loc, double scale);
 double rk_gumbel(rk_state *state, double loc, double scale);
 double rk_logistic(rk_state *state, double loc, double scale);
 double rk_lognormal(rk_state *state, double mean, double sigma);
 double rk_rayleigh(rk_state *state, double mode);
 double rk_wald(rk_state *state, double mean, double scale);
 long rk_zipf(rk_state *state, double a);
 long rk_geometric(rk_state *state, double p);
 long rk_geometric_search(rk_state *state, double p);
 long rk_geometric_inversion(rk_state *state, double p);
 long rk_hypergeometric(rk_state *state, long good, long bad, long sample);
 long rk_hypergeometric_hyp(rk_state *state, long good, long bad, long sample);
 long rk_hypergeometric_hrua(rk_state *state, long good, long bad, long sample);
 double rk_triangular(rk_state *state, double left, double mode, double right);
 long rk_logseries(rk_state *state, double p);
]]

-- Function that should not have a lua wrapper
local doNotWrap = {'rk_seed',
    'rk_altfill',
    'rk_devfill',
    'rk_fill'
}

-- Declare the functions to FFI
ffi.cdef(fun_cdef)

-- Extract the list of function and their types
local funs = {}
local returnType, functionName, paramName, paramType
for line in string.gmatch(fun_cdef, "[^\n]+") do
    -- Split words
    local words = {}
    line:gsub("([^ (,);\n\r]+)", function(c) table.insert(words, c) end)

    -- get the return type (long or double, no multiword, easy) and the name

    local type = words[1]
    -- ignore unsigned, not supported in Torch Tensors
    if type == 'unsigned' then
        table.remove(words, 1)
        type = words[1]
    end
    local name = words[2]
    funs[name] = {
        returnType = type,
        name = name,
        arguments = {}
    }
    -- get the parameters
    -- skip words 3 and 4: they are the 'rkstate *state'
    local j = 5
    while  j + 1 <= #words do
        table.insert(funs[name].arguments, {name = words[j+1], type = words[j]})
        j = j + 2
    end
end

-- Remove not-to-wrap functions
for k,v in pairs(doNotWrap) do
    funs[v] = nil
end

funs['rk_interval'] = {
    name = 'interval',
    arguments = { { name = 'max', type = 'long' } },
    returnType = 'long'
}
funs['rk_randint'] = {
    name = 'randint',
    arguments = { { name = 'low', type = 'int' }, { name = 'high', type = 'int' } },
    returnType = 'int'
}
funs['rk_uniform'] = {
    name = 'uniform',
    arguments = { { name = 'low', type = 'int' }, { name = 'high', type = 'int' } },
    returnType = 'int'
}

--[[ Initialize the state structure (which is not really seeding,
   since we have replaced randomkit's own Mersenne-Twister by
   Torch's ]]
randomkit._state = ffi.new('rk_state')
randomkit.ffi.rk_seed(0, randomkit._state)

-- Extend torch state handling to handle randomkit's state too
local _manualSeed = torch.manualSeed
torch.manualSeed = function(seed)
    randomkit.ffi.rk_seed(0, randomkit._state)
    return _manualSeed(seed)
end

local _getRNGState = torch.getRNGState
torch.getRNGState = function()
    -- Serialize to string, required to write to file
    local clonedState = ffi.string(randomkit._state, ffi.sizeof(randomkit._state))
    return {
        torch = _getRNGState(),
        randomkit = clonedState
    }
end

local _setRNGState = torch.setRNGState
torch.setRNGState = function(state)
    if not type(state) == 'table' or not state.torch or not state.randomkit then
        error('State was not saved with randomkit, cannot set it back')
    end
    _setRNGState(state.torch)
    -- Deserialize from string
    ffi.copy(randomkit._state, state.randomkit, ffi.sizeof(randomkit._state))
end

local returnTypeMapping = {
    int = torch.IntTensor,
    double = torch.DoubleTensor
}
local function getDataArray(tensor)
    local pointerDef = torch.typename(tensor):gfind('torch%.(.*Tensor)')().."*"
    return ffi.cast(pointerDef, torch.pointer(tensor)).storage.data
end
local function generateIntoTensor(output, func)
    if not output:isContiguous() then
        error("generateIntoTensor only supports contiguous tensors")
    end

    local outputdata = getDataArray(output)
    local offset = output:storageOffset()
    -- A zero-based index is used to access the data.
    -- The end index is (startIndex + nElements - 1).
    for i0 = offset - 1, offset - 1 + output:nElement() - 1 do
        outputdata[i0] = tonumber(func(randomkit._state)) or outputdata[i0]
    end
    return output
end
local function applyNotInPlace(input, output, func)
    if not input:isContiguous() or not output:isContiguous() then
        error("applyNotInPlace only supports contiguous tensors")
    end

    if input:nElement() ~= output:nElement() then
        error("applyNotInPlace: tensor element counts are not consistent")
    end

    local inputdata = getDataArray(input)
    local outputdata = getDataArray(output)
    local offset = input:storageOffset()
    -- A zero-based index is used to access the data.
    -- The end index is (startIndex + nElements - 1).
    for i0 = offset - 1, offset - 1 + input:nElement() - 1 do
        outputdata[i0] = tonumber(func(randomkit._state, inputdata[i0])) or outputdata[i0]
    end
    return output
end
local function mapNotInPlace(inputA, inputB, output, func)
    if not inputA:isContiguous() or not inputB:isContiguous() or not output:isContiguous() then
        error("mapNotInPlace only supports contiguous tensors")
    end

    if inputA:nElement() ~= inputB:nElement() or inputB:nElement() ~= output:nElement() then
        error("mapNotInPlace: tensor element counts are not consistent")
    end

    local inputAdata = getDataArray(inputA)
    local inputBdata = getDataArray(inputB)
    local outputdata = getDataArray(output)
    local offset = inputA:storageOffset()
    -- A zero-based index is used to access the data.
    -- The end index is (startIndex + nElements - 1).
    for i0 = offset - 1, offset - 1 + inputA:nElement() - 1 do
        outputdata[i0] = tonumber(func(randomkit._state, inputAdata[i0], inputBdata[i0])) or outputdata[i0]
    end
    return output
end
local function map2NotInPlace(inputA, inputB, inputC, output, func)
    if not inputA:isContiguous() or not inputB:isContiguous() or not inputC:isContiguous() or not output:isContiguous() then
        error("map2NotInPlace only supports contiguous tensors")
    end

    if inputA:nElement() ~= inputB:nElement()
        or inputB:nElement() ~= output:nElement()
        or inputC:nElement() ~= output:nElement() then
        error("map2NotInPlace: tensor element counts are not consistent")
    end

    local inputAdata = getDataArray(inputA)
    local inputBdata = getDataArray(inputB)
    local inputCdata = getDataArray(inputC)
    local outputdata = getDataArray(output)
    local offset = inputA:storageOffset()
    -- A zero-based index is used to access the data.
    -- The end index is (startIndex + nElements - 1).
    for i0 = offset - 1, offset - 1 + inputA:nElement() - 1 do
        outputdata[i0] = tonumber(func(randomkit._state, inputAdata[i0], inputBdata[i0], inputCdata[i0])) or outputdata[i0]
    end
    return output
end

local function create_wrapper(name, randomkitFunction, parameters, returnType)

    -- Note: default to DoubleTensor for arguments we don't know how to deal with
    local tensorReturnType = returnTypeMapping[returnType] or torch.DoubleTensor

    local function help()
        local argNames = ""
        if #parameters > 0 then
            argNames = argNames .. parameters[1].name
        end
        for i = 2,#parameters do
            argNames = argNames .. ", " .. parameters[i].name

        end

        print("Usage: randomkit." .. name .. "(" .. argNames .. ")")
        print("Returns: " .. returnType)
    end

    local function wrapper(...)
        local result, params = randomkit._check1DParams(#parameters, tensorReturnType, ...)

        if result then
            if #params == 0 then
                generateIntoTensor(result, randomkitFunction)
            elseif #params == 1 then
                params[1] = params[1]:contiguous()
                applyNotInPlace(params[1], result, randomkitFunction)
            elseif #params == 2 then
                params[1] = params[1]:contiguous()
                params[2] = params[2]:contiguous()
                mapNotInPlace(params[1], params[2], result, randomkitFunction)
            elseif #params == 3 then
                params[1] = params[1]:contiguous()
                params[2] = params[2]:contiguous()
                params[3] = params[3]:contiguous()
                map2NotInPlace(params[1], params[2], params[3], result, randomkitFunction)
            else
                error('TODO: need to implement map for ' .. #params .. 'arguments')
            end
        else
            result = tonumber(randomkitFunction(randomkit._state, unpack(params)))
        end

        return result
    end
    return wrapper
end

local customWrappers = {}
customWrappers.interval = function(state, max)
    return randomkit.ffi.rk_interval(max, state)
end
customWrappers.randint = function(state, low, high)
    return low + tonumber(randomkit.ffi.rk_interval(high - low, state))
end
customWrappers.uniform = function(state, low, high)
    return randomkit.ffi.rk_uniform(state, low, high - low)
end

-- Wrap by passing the state as first argument
for k, v in pairs(funs) do
    local randomkitFunction
    if customWrappers[v.name] then
        randomkitFunction = customWrappers[v.name]
    else
        randomkitFunction = randomkit.ffi[v.name]
    end
    randomkit[string.sub(k, 4)] =  create_wrapper(v.name, randomkitFunction, v.arguments, v.returnType)
end

