# Unit Tests

This page contains some additional information on torch-distributions's unit tests. For the most part, the tests should be self explanatory, but the Multivariate Gaussian has a lot of cases to test and so is somewhat more involved.

### testMultivariateGaussian.lua

The tests for the PDF and log-PDF are straight-forward, since these functions are deterministic - we just compare the output to a window of reference data. We do this for a standard and one non-standard Gaussian distribution.

Testing the `multivariateGaussianRand` function is complicated by the facts that:

* the function is not deterministic, so we must rely on statistical tests for correctness
* the function supports many different forms of invocation

For this reason, the bulk of the test cases is generated automatically at runtime - by the `generateSystematicTests` function. This constructs tables of possible input arguments, and generates a test case for each combination thereof. Different combinations of inputs lead to different expected results, so there is also a table (`expectations`) mapping from the input form (as encoded in a human-readable string) to a function which checks the result of that function call.

In the cases where the combination of inputs is a valid one, the function that checks the result will perform a statistical test on the returned samples, to determine whether it comes from the correct distribution. However - since the samples are generated randomly, there is a small chance that it may not appear to come from the distribution even if it was generated correctly! In this case our test will fail, incorrectly. Since we perform many statistical tests in the course of running the test suite, the chance of this happening at least once is amplified to an undesirable degree. We therefore compensate for this using Sidak's Correction, which makes it proportionally less likely that each test will fail.
