require 'distributions'
local myTest = {}
local tester = torch.Tester()

function myTest.gaussianpdf()
    tester:assertalmosteq(distributions.norm.pdf(-0.5, 0, 1), 0.352065326764300, 1e-15)
    tester:assertalmosteq(distributions.norm.pdf(0,    0, 1), 0.398942280401433, 1e-15)
    tester:assertalmosteq(distributions.norm.pdf(1,    0, 1), 0.241970724519143, 1e-15)
    tester:assertalmosteq(distributions.norm.pdf(4,    0, 1), 0.000133830225765, 1e-15)
    tester:assertalmosteq(distributions.norm.pdf(-0.5, 2, 3), 0.093970625136768, 1e-15)
    tester:assertalmosteq(distributions.norm.pdf(0,    2, 3), 0.106482668507451, 1e-15)
    tester:assertalmosteq(distributions.norm.pdf(1,    2, 3), 0.125794409230998, 1e-15)
    tester:assertalmosteq(distributions.norm.pdf(4,    2, 3), 0.106482668507451, 1e-15)
end

function myTest.gaussianlogpdf()
    tester:assertalmosteq(distributions.norm.logpdf(-0.5, 0, 1), math.log(0.352065326764300), 1e-12)
    tester:assertalmosteq(distributions.norm.logpdf(0,    0, 1), math.log(0.398942280401433), 1e-12)
    tester:assertalmosteq(distributions.norm.logpdf(1,    0, 1), math.log(0.241970724519143), 1e-12)
    tester:assertalmosteq(distributions.norm.logpdf(4,    0, 1), math.log(0.000133830225765), 1e-12)
    tester:assertalmosteq(distributions.norm.logpdf(-0.5, 2, 3), math.log(0.093970625136768), 1e-12)
    tester:assertalmosteq(distributions.norm.logpdf(0,    2, 3), math.log(0.106482668507451), 1e-12)
    tester:assertalmosteq(distributions.norm.logpdf(1,    2, 3), math.log(0.125794409230998), 1e-12)
    tester:assertalmosteq(distributions.norm.logpdf(4,    2, 3), math.log(0.106482668507451), 1e-12)
end

function myTest.gaussiancdf()
    tester:assertalmosteq(distributions.norm.cdf(-0.5, 0, 1), 0.308537538725987, 1e-15)
    tester:assertalmosteq(distributions.norm.cdf(0,    0, 1), 0.500000000000000, 1e-15)
    tester:assertalmosteq(distributions.norm.cdf(1,    0, 1), 0.841344746068543, 1e-15)
    tester:assertalmosteq(distributions.norm.cdf(4,    0, 1), 0.999968328758167, 1e-15)
    tester:assertalmosteq(distributions.norm.cdf(-0.5, 2, 3), 0.202328380963643, 1e-15)
    tester:assertalmosteq(distributions.norm.cdf(0,    2, 3), 0.252492537546923, 1e-15)
    tester:assertalmosteq(distributions.norm.cdf(1,    2, 3), 0.369441340181764, 1e-15)
    tester:assertalmosteq(distributions.norm.cdf(4,    2, 3), 0.747507462453077, 1e-15)
end

tester:add(myTest)
tester:run()
