local function pks(z)
    assert(z >= 0, 'Wrong z')
    if z == 0 then return 0 end
    if z < 1.18 then
        local y = math.exp(-1.23370055013616983/math.sqrt(z))
        return 2.25675833419102515 * math.sqrt(- math.log(y)) *
            ( y + y^9 + y^25 + y^49)
    else
        local x = math.exp(-2 * math.sqrt(z))
        return 1 - 2*(x - x^4 + x^9)
    end
end

local function qks(z)
    assert(z >= 0, 'Wrong z')
    if z == 0 then return 1 end
    if z < 1.18 then
        return 1 - pks(z)
    end
    local x = math.exp(-2 * math.sqrt(z))
    return 2 * (x - x^4 - x^9)
end

--[[ One-sample Kolmogorov Smirnov test

One-samples Kolmogorov-Smirnov test [1]. Implements the pks and kqs functions
from [2].
[1] http://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
[2] Press, W. H. et al. (2007). Numerical recipes : the art of scientific
    computing. Cambridge, UK; New York: Cambridge University Press.

Arguments:

* `x` (1D Tensor) tensor of samples
* `cdf` (function) function returning the value of CDF of interest at a point

Returns:

1. p-value of the test
2. value of the test statistic
]]
function distributions.ksone(x, cdf)

    local d = 0
    local fo = 0
    local n = x:nElement()
    local en = n

    x = x:sort()
    for j = 1, n do
        local fn = j / en
        local ff = cdf(x[j])
        local dt = math.max(math.abs(fo - ff), math.abs(fn - ff))
        d = math.max(d, dt)
        fo = fn
    end

    en = math.sqrt(en)
    local p = qks( (en + 0.12 + 0.11/en) * d)

    return p, d
end

--[[ Two-sample Kolmogorov Smirnov test

Two-samples Kolmogorov-Smirnov test [1]. Implements the pks and kqs functions
from [2].
[1] http://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov.E2.80.93Smirnov_test
[2] Press, W. H. et al. (2007). Numerical recipes : the art of scientific
    computing. Cambridge, UK; New York: Cambridge University Press.

Arguments:

* `x1` (1D Tensor) tensor of samples
* `x2` (1D Tensor) tensor of samples

Returns:

1. p-value of the test
2. value of the test statistic
]]
function distributions.kstwo(x1, x2)

    local p
    local d = 0
    local j1, j2 = 1, 1
    local fn1, fn2 = 0, 0
    local n1, n2 = x1:nElement(), x2:nElement()
    local en1, en2 = n1, n2

    x1 = x1:sort()
    x2 = x2:sort()
    while j1 <= n1 and j2 <= n2 do
        local d1, d2 = x1[j1], x2[j2]

        if d1 <= d2 then
            while j1 <= n1 and d1 == x1[j1] do
                j1 = j1 + 1
                fn1 = j1 / en1
            end
        end

        if d2 <= d1 then
            while j2 <= n2 and d2 == x2[j2] do
                j2 = j2 + 1
                fn2 = j2 / en2
            end
        end

        local dt = math.abs(fn2 - fn1)
        if dt > d then
            d = dt
        end
    end

    local en = math.sqrt(en1 * en2 / (en1 + en2) )
    local p = qks( (en + 0.12 + 0.11/en)  * d)

    return p, d
end
