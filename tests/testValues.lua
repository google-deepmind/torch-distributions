require 'randomkit'
local ffi = require 'ffi'
require 'torchffi'

local myTest = {}
local tester = torch.Tester()
local seed = 1234567890

function myTest.test_binomial_n_zero()
    -- Tests the corner case of n == 0 for the binomial distribution.
    -- binomial(0, p) should be zero for any p in {0, 1}.
    -- This test addresses issue --3480.
    local zeros = torch.IntTensor(2):zero()
    for _, p in ipairs({0, .5, 1}) do
        tester:asserteq(randomkit.binomial(0, p), 0)
    end
end

--[[ TODO: wrap random_integers or remove this
function myTest.test_randint()
    torch.manualSeed(seed)
    local actual = torch.IntTensor(3, 2):random(-99, 99)
    local desired = torch.Tensor({{ 31,   3},
    {-52,  41},
    {-48, -66}})
    tester:asserteq(actual, desired)
end
--]]

function myTest.test_random_sample()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.random_sample(actual)
    local desired = torch.Tensor({{ 0.61879477158567997,  0.59162362775974664},
    { 0.88868358904449662,  0.89165480011560816},
    { 0.4575674820298663,  0.7781880808593471 }})
    tester:assertTensorEq(actual, desired, 1e-5)
end

--[[ TODO: wrap choice functions or remove these tests
function myTest.test_choice_uniform_replace()
    torch.manualSeed(seed)
    local actual = randomkit.choice(4, 4)
    local desired = torch.Tensor({2, 3, 2, 3})
    tester:asserteq(actual, desired)
end
function myTest.test_choice_nonuniform_replace()
    torch.manualSeed(seed)
    local actual = randomkit.choice(4, 4, {0.4, 0.4, 0.1, 0.1})
    local desired = torch.Tensor({1, 1, 2, 2})
    tester:asserteq(actual, desired)
end
function myTest.test_choice_uniform_noreplace()
    torch.manualSeed(seed)
    local actual = randomkit.choice(4, 3, false)
    local desired = torch.Tensor({0, 1, 3})
    tester:asserteq(actual, desired)
end
function myTest.test_choice_nonuniform_noreplace()
    torch.manualSeed(seed)
    local actual = randomkit.choice(4, 3, false, {0.1, 0.3, 0.5, 0.1})
    local desired = torch.Tensor({2, 3, 1})
    tester:asserteq(actual, desired)
end
function myTest.test_choice_noninteger()
    torch.manualSeed(seed)
    local actual = randomkit.choice({'a', 'b', 'c', 'd'}, 4)
    local desired = torch.Tensor({'c', 'd', 'c', 'd'})
    tester:asserteq(actual, desired)
end
function myTest.test_choice_exceptions()
    sample = randomkit.choice
    tester:assertError(ValueError, sample, -1, 3)
    tester:assertError(ValueError, sample, 3., 3)
    tester:assertError(ValueError, sample, {{1, 2}, {3, 4}}, 3)
    tester:assertError(ValueError, sample, {}, 3)
    tester:assertError(ValueError, sample, {1, 2, 3, 4}, 3, {{0.25, 0.25}, {0.25, 0.25}})
    tester:assertError(ValueError, sample, {1, 2}, 3, {0.4, 0.4, 0.2})
    tester:assertError(ValueError, sample, {1, 2}, 3, {1.1, -0.1})
    tester:assertError(ValueError, sample, {1, 2}, 3, {0.4, 0.4})
    tester:assertError(ValueError, sample, {1, 2, 3}, 4, false)
    tester:assertError(ValueError, sample, {1, 2, 3}, 2, false,
    {1, 0, 0})
end
function myTest.test_choice_return_shape()
    p = {0.1, 0.9}
    -- Check scalar
    tester:assert(isscalar(randomkit.choice(2, true)))
    tester:assert(isscalar(randomkit.choice(2, false)))
    tester:assert(isscalar(randomkit.choice(2, true, p)))
    tester:assert(isscalar(randomkit.choice(2, false, p)))
    tester:assert(isscalar(randomkit.choice({1, 2}, true)))
    tester:asserteq(randomkit.choice({None}, true), nil)
    local a = torch.Tensor({1, 2})
    local arr = torch.Tensor(1)
    arr[1] = a
    tester:assertTensorEq(randomkit.choice(arr, true), a)

    -- Check 0-d torch.Tensor
    s = tuple()
    tester:assert(not isscalar(randomkit.choice(2, s, true)))
    tester:assert(not isscalar(randomkit.choice(2, s, false)))
    tester:assert(not isscalar(randomkit.choice(2, s, true, p)))
    tester:assert(not isscalar(randomkit.choice(2, s, false, p)))
    tester:assert(not isscalar(randomkit.choice({1, 2}, s, true)))
    tester:assert(randomkit.choice({None}, s, true).ndim == 0)
    a = torch.Tensor({1, 2})
    local arr = torch.Tensor(1)
    arr[0] = a
    tester:assertTensorEq(randomkit.choice(arr, s, true).item(), a)

    -- Check multi dimensional torch.Tensor
    s = {2, 3}
    p = {0.1, 0.1, 0.1, 0.1, 0.4, 0.2}
    tester:assert(randomkit.choice(6, s, true).shape, s)
    tester:assert(randomkit.choice(6, s, false).shape, s)
    tester:assert(randomkit.choice(6, s, true, p).shape, s)
    tester:assert(randomkit.choice(6, s, false, p).shape, s)
    tester:assert(randomkit.choice(arange(6), s, true).shape, s)
end
--]]
function myTest.test_bytes()
    torch.manualSeed(seed)
    local actual = randomkit.bytes(torch.ByteTensor(10)):double()
    local desired = torch.ByteTensor({130, 85, 105, 158, 255, 151, 43, 87, 102, 165}):double()
    tester:assertTensorEq(actual, desired, 1e-15, "Bytes sampler doesn't produce desired values")
end

function myTest.test_beta()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.beta(actual, .1, .9)
    local desired = torch.Tensor({{  1.45341850513746058e-02,   5.31297615662868145e-04},
    {  1.85366619058432324e-06,   4.19214516800110563e-03},
    {  1.58405155108498093e-04,   1.26252891949397652e-04}})
    tester:assertTensorEq(actual, desired, 1e-15, "Beta sampler doesn't produce desired values")
end
function myTest.test_binomial()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.binomial(actual, 100.123, .456)
    local desired = torch.Tensor({{37, 43},
    {42, 48},
    {46, 45}})
    tester:assertTensorEq(actual, desired, 1e-15, "Binomial sampler doesn't produce desired values")
end
function myTest.test_chisquare()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.chisquare(actual, 50)
    local desired = torch.Tensor({{ 63.87858175501090585,  68.68407748911370447},
    { 65.77116116901505904,  47.09686762438974483},
    { 72.3828403199695174,  74.18408615260374006}})
    tester:assertTensorEq(actual, desired, 1e-13, "Chi-square sampler doesn't produce desired values")
end
function myTest.test_exponential()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.exponential(actual, 1.1234)
    local desired = torch.Tensor({{ 1.08342649775011624,  1.00607889924557314},
    { 2.46628830085216721,  2.49668106809923884},
    { 0.68717433461363442,  1.69175666993575979}})
    tester:assertTensorEq(actual, desired, 1e-15, "Exponential sampler doesn't produce desired values")
end
function myTest.test_f()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.f(actual, 12, 77)
    local desired = torch.Tensor({{ 1.21975394418575878,  1.75135759791559775},
    { 1.44803115017146489,  1.22108959480396262},
    { 1.02176975757740629,  1.34431827623300415}})
    tester:assertTensorEq(actual, desired, 1e-15, "F sampler doesn't produce desired values")
end
function myTest.test_gamma()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.gamma(actual, 5, 3)
    local desired = torch.Tensor({{ 24.60509188649287182,  28.54993563207210627},
    { 26.13476110204064184,  12.56988482927716078},
    { 31.71863275789960568,  33.30143302795922011}})
    tester:assertTensorEq(actual, desired, 1e-14, "Gamma sampler doesn't produce desired values")
end
function myTest.test_geometric()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.geometric(actual, .123456789)
    local desired = torch.Tensor({{ 8,  7},
    {17, 17},
    { 5, 12}})
    tester:assertTensorEq(actual, desired, 1e-15, "Geometric sampler doesn't produce desired values")
end
function myTest.test_gumbel()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.gumbel(actual, .123456789, 2.0)
    local desired = torch.Tensor({{ 0.19591898743416816,  0.34405539668096674},
    {-1.4492522252274278, -1.47374816298446865},
    { 1.10651090478803416, -0.69535848626236174}})
    tester:assertTensorEq(actual, desired, 1e-15, "Gumbel sampler doesn't produce desired values")
end
function myTest.test_hypergeometric()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.hypergeometric(actual, 10.1, 5.5, 14)
    local desired = torch.Tensor({{10, 10},
    {10, 10},
    { 9,  9}})
    tester:assertTensorEq(actual, desired, 1e-15, "Hypergeometric sampler doesn't produce desired values")

    -- Test nbad = 0
    actual = torch.Tensor(4)
    randomkit.hypergeometric(actual, 5, 0, 3)
    desired = torch.Tensor({3, 3, 3, 3})
    tester:assertTensorEq(actual, desired, 1e-15, "Hypergeometric sampler doesn't produce desired values")

    randomkit.hypergeometric(actual, 15, 0, 12)
    desired = torch.Tensor({12, 12, 12, 12})
    tester:assertTensorEq(actual, desired, 1e-15, "Hypergeometric sampler doesn't produce desired values")

    -- Test ngood = 0
    randomkit.hypergeometric(actual, 0, 5, 3)
    desired = torch.Tensor({0, 0, 0, 0})
    tester:assertTensorEq(actual, desired, 1e-15, "Hypergeometric sampler doesn't produce desired values")

    randomkit.hypergeometric(actual, 0, 15, 12)
    desired = torch.Tensor({0, 0, 0, 0})
    tester:assertTensorEq(actual, desired, 1e-15, "Hypergeometric sampler doesn't produce desired values")
end
function myTest.test_laplace()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.laplace(actual, .123456789, 2.0)
    local desired = torch.Tensor({{ 0.66599721112760157,  0.52829452552221945},
    { 3.12791959514407125,  3.18202813572992005},
    {-0.05391065675859356,  1.74901336242837324}})
    tester:assertTensorEq(actual, desired, 1e-15, "Laplace sampler doesn't produce desired values")
end
function myTest.test_logistic()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.logistic(actual, .123456789, 2.0)
    local desired = torch.Tensor({{ 1.09232835305011444,  0.8648196662399954 },
    { 4.27818590694950185,  4.33897006346929714},
    {-0.21682183359214885,  2.63373365386060332}})
    tester:assertTensorEq(actual, desired, 1e-15, "Logistic sampler doesn't produce desired values")
end
function myTest.test_lognormal()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.lognormal(actual, .123456789, 2.0)
    local desired = torch.Tensor({{ 16.50698631688883822,  36.54846706092654784},
    { 22.67886599981281748,   0.71617561058995771},
    { 65.72798501792723869,  86.84341601437161273}})
    tester:assertTensorEq(actual, desired, 1e-13, "Log-normal sampler doesn't produce desired values")
end
function myTest.test_logseries()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.logseries(actual, .923456789)
    local desired = torch.Tensor({{ 2,  2},
    { 6, 17},
    { 3,  6}})
    tester:assertTensorEq(actual, desired, 1e-14, "Log-series sampler doesn't produce desired values")
end
function myTest.test_negative_binomial()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.negative_binomial(actual, 100, .12345)
    local desired = torch.Tensor({{848, 841},
    {892, 611},
    {779, 647}})
    tester:assertTensorEq(actual, desired, 1e-15, "Negative binomial sampler doesn't produce desired values")
end
function myTest.test_noncentral_chisquare()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.noncentral_chisquare(actual, 5, 5)
    local desired = torch.Tensor({{ 23.91905354498517511,  13.35324692733826346},
    { 31.22452661329736401,  16.60047399466177254},
    {  5.03461598262724586,  17.94973089023519464}})
    tester:assertTensorEq(actual, desired, 1e-14, "Non-central Chi-square sampler doesn't produce desired values")
end

function myTest.test_noncentral_f()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.noncentral_f(actual, 5, 2, 1)
    local desired = torch.Tensor({{ 1.40598099674926669,  0.34207973179285761},
    { 3.57715069265772545,  7.92632662577829805},
    { 0.43741599463544162,  1.1774208752428319 }})
    tester:assertTensorEq(actual, desired, 1e-14, "Non-central F sampler doesn't produce desired values")
end
function myTest.test_normal()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.normal(actual, .123456789, 2.0)
    local desired = torch.Tensor({{ 2.80378370443726244,  3.59863924443872163},
    { 3.121433477601256, -0.33382987590723379},
    { 4.18552478636557357,  4.46410668111310471}})
    tester:assertTensorEq(actual, desired, 1e-15, "Gaussian sampler doesn't produce desired values")
end
function myTest.test_pareto()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.pareto(actual, .123456789)
    local desired = torch.Tensor({{  2.46852460439034849e+03,   1.41286880810518346e+03},
    {  5.28287797029485181e+07,   6.57720981047328785e+07},
    {  1.40840323350391515e+02,   1.98390255135251704e+05}})
    tester:assertTensorEq(actual, desired, 1e-15, "Pareto sampler doesn't produce desired values")
end
function myTest.test_poisson()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.poisson(actual, .123456789)
    local desired = torch.Tensor({{0, 0},
    {1, 0},
    {0, 0}})
    tester:assertTensorEq(actual, desired, 1e-15, "Poisson sampler doesn't produce desired values")
end
--[[ TODO enable error handling
function myTest.test_poisson_exceptions()
    lambig = iinfo('l').max
    lamneg = -1
    tester:assertError(ValueError, randomkit.poisson, lamneg)
    tester:assertError(ValueError, randomkit.poisson, {lamneg}*10)
    tester:assertError(ValueError, randomkit.poisson, lambig)
    tester:assertError(ValueError, randomkit.poisson, {lambig}*10)
end
--]]
function myTest.test_power()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.power(actual, .123456789)
    local desired = torch.Tensor({{ 0.02048932883240791,  0.01424192241128213},
    { 0.38446073748535298,  0.39499689943484395},
    { 0.00177699707563439,  0.13115505880863756}})
    tester:assertTensorEq(actual, desired, 1e-15, "Power sampler doesn't produce desired values")
end
function myTest.test_rayleigh()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.rayleigh(actual, 10)
    local desired = torch.Tensor({{ 13.8882496494248393,  13.383318339044731  },
    { 20.95413364294492098,  21.08285015800712614},
    { 11.06066537006854311,  17.35468505778271009}})
    tester:assertTensorEq(actual, desired, 1e-14, "Rayleigh sampler doesn't produce desired values")
end
function myTest.test_standard_cauchy()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.standard_cauchy(actual)
    local desired = torch.Tensor({{ 0.77127660196445336, -6.55601161955910605},
    { 0.93582023391158309, -2.07479293013759447},
    {-4.74601644297011926,  0.18338989290760804}})
    tester:assertTensorEq(actual, desired, 1e-15)
end
function myTest.test_standard_exponential()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.standard_exponential(actual)
    local desired = torch.Tensor({{ 0.96441739162374596,  0.89556604882105506},
    { 2.1953785836319808,  2.22243285392490542},
    { 0.6116915921431676,  1.50592546727413201}})
    tester:assertTensorEq(actual, desired, 1e-15)
end
function myTest.test_standard_gamma()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.standard_gamma(actual, 3)
    local desired = torch.Tensor({{ 5.50841531318455058,  6.62953470301903103},
    { 5.93988484943779227,  2.31044849402133989},
    { 7.54838614231317084,  8.012756093271868  }})
    tester:assertTensorEq(actual, desired, 1e-14, "Standard Gamma sampler doesn't produce desired values")

end
--[[ TODO find this!
function myTest.test_standard_normal()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.standard_normal(actual)
    local desired = torch.Tensor({{ 1.34016345771863121,  1.73759122771936081},
    { 1.498988344300628, -0.2286433324536169 },
    { 2.031033998682787,  2.17032494605655257}})
    tester:assertTensorEq(actual, desired, 1e-15)
end
--]]
function myTest.test_standard_t()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.standard_t(actual, 10)
    local desired = torch.Tensor({{ 0.97140611862659965, -0.08830486548450577},
    { 1.36311143689505321, -0.55317463909867071},
    {-0.18473749069684214,  0.61181537341755321}})
    tester:assertTensorEq(actual, desired, 1e-15, "Standard T sampler doesn't produce desired values")

end
function myTest.test_triangular()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.triangular(actual, 5.12, 10.23, 20.34)
    local desired = torch.Tensor({{ 12.68117178949215784,  12.4129206149193152 },
    { 16.20131377335158263,  16.25692138747600524},
    { 11.20400690911820263,  14.4978144835829923 }})
    tester:assertTensorEq(actual, desired, 1e-14, "Triangular sampler doesn't produce desired values")
end
--[[ TODO work out why this fails
function myTest.test_uniform()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.uniform(actual, 1.23, 10.54)
    local desired = torch.Tensor({{ 6.99097932346268003,  6.73801597444323974},
    { 9.50364421400426274,  9.53130618907631089},
    { 5.48995325769805476,  8.47493103280052118}})
    print(actual, desired)
    tester:assertTensorEq(actual, desired, 1e-15)
end
--]]
function myTest.test_vonmises()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.vonmises(actual, 1.23, 1.54)
    local desired = torch.Tensor({{ 2.28567572673902042,  2.89163838442285037},
    { 0.38198375564286025,  2.57638023113890746},
    { 1.19153771588353052,  1.83509849681825354}})
    tester:assertTensorEq(actual, desired, 1e-15, "Von Mises sampler doesn't produce desired values")

end
function myTest.test_wald()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.wald(actual, 1.23, 1.54)
    local desired = torch.Tensor({{ 3.82935265715889983,  5.13125249184285526},
    { 0.35045403618358717,  1.50832396872003538},
    { 0.24124319895843183,  0.22031101461955038}})
    tester:assertTensorEq(actual, desired, 1e-14, "Wald sampler doesn't produce desired values")

end
function myTest.test_weibull()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.weibull(actual, 1.23)
    local desired = torch.Tensor({{ 0.97097342648766727,  0.91422896443565516},
    { 1.89517770034962929,  1.91414357960479564},
    { 0.67057783752390987,  1.39494046635066793}})
    tester:assertTensorEq(actual, desired, 1e-15, "Weibull sampler doesn't produce desired values")

end
function myTest.test_zipf()
    torch.manualSeed(seed)
    local actual = torch.Tensor(3, 2)
    randomkit.zipf(actual, 1.23)
    local desired = torch.Tensor({{66, 29},
    { 1,  1},
    { 3, 13}})
    tester:assertTensorEq(actual, desired, 1e-15, "Zipf sampler doesn't produce desired values")
end

function myTest.test_returnType()
    tester:asserteq(type(randomkit.binomial(10, 0.5)), 'number')
end

tester:add(myTest)
tester:run()
