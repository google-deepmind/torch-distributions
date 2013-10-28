local ffi = require("ffi")

randomkit = {}
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

--[[ Initialize the state structure (which is not really seeding,
   since we have replaced randomkit's own Mersenne-Twister by
   Torch's ]]
local state = ffi.new('rk_state')
randomkit.ffi.rk_seed(0, state)

-- Wrap by passing the state as first argument
for k, v in pairs(funs) do
    randomkit[string.sub(k, 4)] =  function(...)
        return tonumber(randomkit.ffi[k](state, ...))
    end
end

return randomkit

