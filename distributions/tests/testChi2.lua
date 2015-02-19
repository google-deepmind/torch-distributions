require 'totem'
require 'distributions'
local myTest = {}
local tester = totem.Tester()

function myTest.chi2PDF()
    tester:assert(cephes.isinf(distributions.chi2.pdf(0, 1)))
    tester:assertalmosteq(distributions.chi2.pdf(1, 1), 0.241970724519143, 1e-15)
    tester:assertalmosteq(distributions.chi2.pdf(2, 1), 0.103776874355149, 1e-15)
    tester:assertalmosteq(distributions.chi2.pdf(5, 1), 0.014644982561926, 1e-15)
    tester:assertalmosteq(distributions.chi2.pdf(0, 9), 0                , 1e-15)
    tester:assertalmosteq(distributions.chi2.pdf(1, 9), 0.002304483090659, 1e-15)
    tester:assertalmosteq(distributions.chi2.pdf(2, 9), 0.015813618949356, 1e-15)
    tester:assertalmosteq(distributions.chi2.pdf(5, 9), 0.087172515249562, 1e-15)
end

function myTest.chi2LogPDF()
    tester:assertalmosteq(distributions.chi2.logpdf(1, 1),-1.418938533204673, 1e-15)
    tester:assertalmosteq(distributions.chi2.logpdf(2, 1),-2.265512123484645, 1e-15)
    tester:assertalmosteq(distributions.chi2.logpdf(5, 1),-4.223657489421723, 1e-15)
    tester:assertalmosteq(distributions.chi2.logpdf(1, 9),-6.072898883362196, 1e-15)
    tester:assertalmosteq(distributions.chi2.logpdf(2, 9),-4.146883751402387, 1e-15)
    tester:assertalmosteq(distributions.chi2.logpdf(5, 9),-2.439866189842844, 1e-15)
end

function myTest.chi2CDF()
    tester:assertalmosteq(distributions.chi2.cdf(0, 1), 0                , 1e-15)
    tester:assertalmosteq(distributions.chi2.cdf(1, 1), 0.682689492137086, 1e-15)
    tester:assertalmosteq(distributions.chi2.cdf(2, 1), 0.842700792949715, 1e-15)
    tester:assertalmosteq(distributions.chi2.cdf(5, 1), 0.974652681322532, 1e-15)
    tester:assertalmosteq(distributions.chi2.cdf(0, 9), 0                , 1e-15)
    tester:assertalmosteq(distributions.chi2.cdf(1, 9), 0.000562497302168, 1e-15)
    tester:assertalmosteq(distributions.chi2.cdf(2, 9), 0.008532393371186, 1e-15)
    tester:assertalmosteq(distributions.chi2.cdf(5, 9), 0.165691739806592, 1e-15)
end

tester:add(myTest)
return tester:run()
