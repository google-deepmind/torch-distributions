require 'distributions'
require 'totem'
local myTest = {}
local tester = totem.Tester()

function myTest.cauchyPDF()
    tester:assertalmosteq(distributions.cauchy.pdf(-0.5, 0, 1), 0.254648, 1e-5)
    tester:assertalmosteq(distributions.cauchy.pdf(0,    0, 1), 1.0/math.pi, 1e-5)
    tester:assertalmosteq(distributions.cauchy.pdf(1,    0, 1), 0.5/math.pi, 1e-5)
    tester:assertalmosteq(distributions.cauchy.pdf(4,    0, 1), 0.0187241, 1e-5)

    tester:assertalmosteq(distributions.cauchy.pdf(-0.5, 2, 3), 0.062618, 1e-5)
    tester:assertalmosteq(distributions.cauchy.pdf(1,    2, 3), 0.095493, 1e-5)
    tester:assertalmosteq(distributions.cauchy.pdf(2,    2, 3), 0.106103, 1e-5)
    tester:assertalmosteq(distributions.cauchy.pdf(4,    2, 3), 0.0734561, 1e-5)
end

function myTest.cauchyLogPDF()
    tester:assertalmosteq(distributions.cauchy.logpdf(-0.5, 0, 1), math.log(0.254648), 1e-5)
    tester:assertalmosteq(distributions.cauchy.logpdf(0,    0, 1), math.log(1.0/math.pi), 1e-5)
    tester:assertalmosteq(distributions.cauchy.logpdf(1,    0, 1), math.log(0.5/math.pi), 1e-5)
    tester:assertalmosteq(distributions.cauchy.logpdf(4,    0, 1), math.log(0.0187241), 1e-5)
    tester:assertalmosteq(distributions.cauchy.logpdf(-0.5, 2, 3), math.log(0.062618), 1e-5)
    tester:assertalmosteq(distributions.cauchy.logpdf(1,    2, 3), math.log(0.095493), 1e-5)
    tester:assertalmosteq(distributions.cauchy.logpdf(2,    2, 3), math.log(0.106103), 1e-5)
    tester:assertalmosteq(distributions.cauchy.logpdf(4,    2, 3), math.log(0.0734561), 1e-5)
end

function myTest.cauchyCDF()
    tester:assertalmosteq(distributions.cauchy.cdf(-0.5, 0, 1), 0.352416, 1e-5)
    tester:assertalmosteq(distributions.cauchy.cdf(0,    0, 1), 0.5, 1e-5)
    tester:assertalmosteq(distributions.cauchy.cdf(1,    0, 1), 0.75, 1e-5)
    tester:assertalmosteq(distributions.cauchy.cdf(4,    0, 1), 0.922021, 1e-5)

    tester:assertalmosteq(distributions.cauchy.cdf(-0.5, 2, 3), 0.278858, 1e-5)
    tester:assertalmosteq(distributions.cauchy.cdf(1,    2, 3), 0.397584, 1e-5)
    tester:assertalmosteq(distributions.cauchy.cdf(2,    2, 3), 0.5, 1e-5)
    tester:assertalmosteq(distributions.cauchy.cdf(4,    2, 3), 0.687167, 1e-5)
end

tester:add(myTest)
return tester:run()
