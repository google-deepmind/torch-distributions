require 'totem'
require 'distributions'
local myTest = {}
local tester = totem.Tester()

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

function myTest.gaussianqtl()
  tester:assertalmosteq(distributions.norm.qtl(0.0000001, 0, 1), -5.199337582187471,   1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.00001,   0, 1), -4.264890793922602,   1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.001,     0, 1), -3.090232306167813,   1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.05,      0, 1), -1.6448536269514729,  1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.15,      0, 1), -1.0364333894937896,  1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.25,      0, 1), -0.6744897501960817,  1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.35,      0, 1), -0.38532046640756773, 1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.45,      0, 1), -0.12566134685507402, 1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.5,       0, 1),  0,                   1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.55,      0, 1),  0.12566134685507402, 1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.65,      0, 1),  0.38532046640756773, 1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.75,      0, 1),  0.6744897501960817,  1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.85,      0, 1),  1.0364333894937896,  1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.95,      0, 1),  1.6448536269514729,  1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.999,     0, 1),  3.090232306167813,   1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.99999,   0, 1),  4.264890793922602,   1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.9999999, 0, 1),  5.199337582187471,   1e-9)

  tester:assertalmosteq(distributions.norm.qtl(0.1,       1, 3), -2.844654696633802,   1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.25,      1, 3), -1.023469250588245,   1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.5,       1, 3),  1,                   1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.75,      1, 3),  3.023469250588245,   1e-9)
  tester:assertalmosteq(distributions.norm.qtl(0.9,       1, 3),  4.844654696633802,   1e-9)
end

tester:add(myTest)
return tester:run()
