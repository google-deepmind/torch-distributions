package = 'distributions'
version = '0-0'

source = {
   url = 'git://github.com/jucor/torch-distributions.git',
   branch = 'master'
}

description = {
  summary = "Probability distributions",
  homepage = "http://jucor.github.io/torch-distributions",
  detailed = "Provides the PDF, CDF, random number generators, and more, for range of probability distributions. It is to torch-randomkit what scipy.stats is to numpy.random.",
  license = "BSD",
  maintainer = "Julien Cornebise <julien@cornebise.com>"
}

dependencies = { 'torch >= 7.0', 'randomkit', 'penlight'}
build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
