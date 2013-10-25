CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

SET(INSTALL_DOCUMENTATION OFF CACHE BOOL "Install the description in Torch tree")

# libcephes compilation and installation is in its own subdirectory
ADD_SUBDIRECTORY(randomkit)

# install the lua code for the cephes package
FILE(GLOB luasrc "*.lua")
ADD_TORCH_PACKAGE(randomkit "${src}" "${luasrc}")
TARGET_LINK_LIBRARIES(randomkit TH)

IF (INSTALL_DOCUMENTATION)
    SUBDIRS(doc)
ENDIF()
#SUBDIRS(tests)
