""" This script tries to generate docs and call tests for randomkit.* """

import scipy
import re

docFile = "doc/randomkit.html"
testFile = "tests/testCalls.lua"
exclude = ['ffi', '_check1DParams']
randomkitFuncsPath = '/Users/daniel.horgan/randomkit_funcs'

def funcTest(name, sig, doc):
    match = re.match(r"(.*)\((.*)\)", sig)
    func = match.group(1)
    args = [x for x in match.group(2).split(",") if x.strip()]
    numArgs = len(args)

    yield """function myTests.test_%s()""" % (name,)
    testArgs = ["0.5"] * (numArgs - 1)
    if name == 'zipf':
        testArgs = ["1.5"] * (numArgs - 1)

    yield """   tester:assert(%s(%s))""" % (func, ", ".join(testArgs))
    testArgs = ["torch.Tensor(10)"] + testArgs
    yield """   tester:assert(%s(%s))""" % (func, ", ".join(testArgs))
    testArgs = ["torch.Tensor(10):fill(0.5)"] * (numArgs - 1)
    if name == 'zipf':
        testArgs = ["torch.Tensor(10):fill(1.5)"] * (numArgs - 1)
    yield """   tester:assert(%s(%s))""" % (func, ", ".join(testArgs))
    testArgs = ["torch.Tensor(10)"] + testArgs
    yield """   tester:assert(%s(%s))""" % (func, ", ".join(testArgs))
    testArgs = ["torch.Tensor(10)"] + testArgs
    yield """   tester:assertError(function() %s(%s) end)""" % (func, ", ".join(testArgs))
    yield """end"""

def funcDoc(name, sig, doc):
    yield "<hr /><a id='%s'>" % (name,)
    yield "<h2>%s</h2>" % (sig,)
    yield "<pre>"
    yield doc
    yield "</pre>"

def genIndex(funcNames):
    index = "<h1>torch-randomkit</h1><ul>"
    for funcName in funcNames:
        index += "<li><a href='#%s'>%s</a></li>" % (funcName, funcName)
    index += "</ul>"
    return index

def funcNames():
    with open(randomkitFuncsPath, 'r') as f:
        for l in f.readlines():
            yield l.strip()

def getDocStrings(funcNames):

    for funcName in funcNames:
        func = getattr(scipy.random, funcName, None)
        if not func:
            print("Could not find scipy docstring for %s" % (funcName,))
            continue

        docLines = func.__doc__.strip().split("\n")
        funcSig = re.sub("=[^,)]+", "", docLines[0])
        funcSig = re.sub(",?\s*size", "", funcSig)
        funcSig = re.sub("\(", "([output], ", funcSig)
        funcSig = "randomkit." + funcSig
        doc = "\n".join(x.strip() for x in docLines[1:])

        yield funcName, funcSig, doc

def writeHTMLdoc(funcNames, funcInfo):
    with open(docFile, 'w') as f:

        f.write("<html>")

        index = genIndex(funcNames)

        f.write(index)

        for name, sig, doc in funcInfo:
            for line in funcDoc(name, sig, doc):
                f.write(line)
            print("Generated doc for " + name)

        f.write("</html>")

def writeCallTests(funcNames, funcInfo):
    with open(testFile, 'w') as f:
        f.write("""
require 'randomkit'
local myTests = {}
local tester = torch.Tester()
""")
        for name, sig, doc in funcInfo:
            for line in funcTest(name, sig, doc):
                f.write(line + "\n")
            print("Generated tests for " + name)

        f.write("""
tester:add(myTests)
tester:run()
""")

funcNames = sorted(list(set(funcNames()) - set(exclude)))

funcInfo = list(getDocStrings(funcNames))

writeHTMLdoc(funcNames, funcInfo)
writeCallTests(funcNames, funcInfo)
