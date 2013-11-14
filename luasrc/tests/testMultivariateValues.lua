
--[[ TODO - wrap multinomial or remove these
function myTest.test_multinomial_basic()
    distributions.multinomial(100, {0.2, 0.8})
end

function myTest.test_multinomial_zero_probability()
    distributions.multinomial(100, {0.2, 0.8, 0.0, 0.0, 0.0})
end

function myTest.test_multinomial_int_negative_interval()
    tester:assert( -5 <= distributions.randint(-5, -1) < -1)
    x = distributions.randint(-5, -1, 5)
    tester:assert(all(-5 <= x))
    tester:assert(all(x < -1))
end
--]]

--[[ TODO - wrap dirichlet or remove this
function myTest.test_dirichlet()
    torch.manualSeed(seed)
    alpha = torch.Tensor({51.72840233779265162,  39.74494232180943953})
    local actual = torch.Tensor(3, 2)
    distributions.dirichlet(actual, alpha)
    local desired = torch.Tensor({{{ 0.54539444573611562,  0.45460555426388438},
    { 0.62345816822039413,  0.37654183177960598}},
    {{ 0.55206000085785778,  0.44793999914214233},
    { 0.58964023305154301,  0.41035976694845688}},
    {{ 0.59266909280647828,  0.40733090719352177},
    { 0.56974431743975207,  0.43025568256024799}}})
    tester:assertTensorEq(actual, desired, 1e-15)
end
--]]
--[[ TODO: wrap multinomial or remove this
function myTest.test_multinomial()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    distributions.multinomial(actual, 20, {1/6, 1/6, 1/6, 1/6, 1/6, 1/6})
    local desired = torch.Tensor({{{4, 3, 5, 4, 2, 2},
    {5, 2, 8, 2, 2, 1}},
    {{3, 4, 3, 6, 0, 4},
    {2, 1, 4, 3, 6, 4}},
    {{4, 4, 2, 5, 2, 3},
    {4, 3, 4, 2, 3, 4}}})
    tester:assertTensorEq(actual, desired)
end
--]]

--[[ TODO: support for multivariate distributions
function myTest.test_multivariate_normal()
torch.manualSeed(seed)
local actual = torch.Tensor(3, 2)
local mean= (.123456789, 10)
local cov = {{1, 0}, {1, 0}}
local size = {3, 2}
local actual = distributions.multivariate_normal(mean, cov, size)
local desired = torch.Tensor({{{ -1.47027513018564449,  10.                 },
{ -1.65915081534845532,  10.                 }},
{{ -2.29186329304599745,  10.                 },
{ -1.77505606019580053,  10.                 }},
{{ -0.54970369430044119,  10.                 },
{  0.29768848031692957,  10.                 }}})
tester:assertTensorEq(actual, desired, 1e-15)
end
--]]

