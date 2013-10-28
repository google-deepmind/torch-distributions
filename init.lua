local ffi = require("ffi")

randomkit = {}
randomkit.ffi = ffi.load(package.searchpath('librandomkit', package.cpath))

local function isTensor(v)
    if torch.typename(v) then
        return string.sub(torch.typename(v), -6, -1) == "Tensor"
    end
end

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

--[[ Initialize the state structure (which is not really seeding,
   since we have replaced randomkit's own Mersenne-Twister by
   Torch's ]]
local state = ffi.new('rk_state')
randomkit.ffi.rk_seed(0, state)

local returnTypeMapping = {
    int = torch.IntTensor,
    double = torch.DoubleTensor
}
local function getDataArray(tensor)
    local pointerDef = torch.typename(tensor):gfind('torch%.(.*Tensor)')().."*"
    return ffi.cast(pointerDef, torch.pointer(tensor)).storage.data
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
        outputdata[i0] = tonumber(func(state, inputdata[i0])) or outputdata[i0]
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
        outputdata[i0] = tonumber(func(state, inputAdata[i0], inputBdata[i0])) or outputdata[i0]
    end
    return output
end


--[[! Argument checking for vectorized calls

Process the optional return storage, the sizes of the parameter functions, etc

@param K number of actual parameters for the sampler
@param defaultResultType Tensor class corresponding to the expected result type (e.g. torch.DoubleTensor, torch.IntegerTensor, etc)
@param ... List of all parameters passed to the original caller

@return T vector or 1-d tensor to store the result into, N rows (or nil, if we should return a single value)
@return p1 ... pk Tensor of parameters, all N rows
--]]
function randomkit._check1DParams(K, defaultResultType, ...)
    local params = { ... }
    if #params ~= K and #params ~= K+1 then
        error('CHKPARAMS: need ' .. K .. ' arguments and optionally, one result tensor, instead got ' .. #params .. ' arguments')
    end

    local result
    local Nresult = nil -- Default: unknown result size
    if #params == K then
        local numberOnly = true
        for paramIndex, param in ipairs(params) do
            numberOnly = numberOnly and not isTensor(param)
        end
        if numberOnly then
            return nil, params
        else
            result = defaultResultType.new(1)
        end
    else
        if isTensor(params[1]) then
            -- The tensor dictates the size of the result
            result = params[1]
            Nresult = result:nElement()
        else
            error("Invalid type " .. type(params[1]) .. " for result")
        end
        table.remove(params, 1)
    end

    -- Ensure that all parameters agree in size
    local Nparams = 1
    for paramIndex, param in ipairs(params) do
        local size
        if isTensor(param) then
            size = param:size(1)
        elseif type(param) == 'number' or type(param) == 'cdata' then
            size = 1
            -- Use torch's default Tensor for parameters
            params[paramIndex] = torch.Tensor{ param }
        else
            error("Invalid type " .. type(param) .. " for parameter " .. paramIndex .. ".")
        end

        if not (size == 1 or Nparams == 1 or Nparams == size) then
            error("Incoherent sizes for parameters")
        elseif size > 1 and Nparams == 1 then
            Nparams = size
        end
    end

    if Nresult then
        -- If the result size was fixed by the caller (either via tensor or integer)
        if Nparams == 1 then
            -- If only size-1 parameters, Nresult dictates the output size
            Nparams = Nresult
        else
            -- However, if the parameters dictate one size and the result another, error
            assert(Nparams == Nresult,  "Parameter size (" .. Nparams ..") does not match result size (" .. Nresult ..")" )
        end
    else
        -- If the result size was not fixed by the caller, parameters dictate it
        Nresult = Nparams
        result:resize(Nresult)
    end

    for paramIndex, param in ipairs(params) do
        if param:size(1) == 1 then
            local sizes = param:size()
            sizes[1] = Nparams
            params[paramIndex] = params[paramIndex]:expand(sizes)
        end
    end

    return result, params
end

local function create_wrapper(name, parameters, returnType)

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
        local argCount = select("#", ...)
        for index = 1,argCount do
            if select(index, ...) == nil then
                error("Bad randomkit call - argument " .. index .. " is nil when calling function " .. name .. "!")
            end
        end
        local result, params = randomkit._check1DParams(#parameters, tensorReturnType, ...)

        if result then
            local randomkitFunction = randomkit.ffi[name]
            if #params == 1 then
                params[1] = params[1]:contiguous()
                applyNotInPlace(params[1], result, randomkitFunction)
            elseif #params == 2 then
                params[1] = params[1]:contiguous()
                params[2] = params[2]:contiguous()
                mapNotInPlace(params[1], params[2], result, randomkitFunction)
            else
                error('TODO: need to implement map for ' .. #params .. 'arguments')
            end
        else
            result = randomkit.ffi[name](state, unpack(params))
        end

        return result
    end
    return wrapper
end


-- Wrap by passing the state as first argument
for k, v in pairs(funs) do
    randomkit[string.sub(k, 4)] =  create_wrapper(v.name, v.arguments, v.returnType)
end

return randomkit

