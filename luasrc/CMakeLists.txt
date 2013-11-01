# install the lua code for the cephes package
FILE(GLOB luasrc "*.lua")
ADD_TORCH_PACKAGE(randomkit "" "${luasrc}")

