FILE(GLOB src "*.c")
ADD_TORCH_PACKAGE(randomkit "${src}" "")
TARGET_LINK_LIBRARIES(randomkit TH)
