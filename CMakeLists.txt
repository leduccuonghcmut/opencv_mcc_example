cmake_minimum_required(VERSION 3.10)
project(opencv_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER "/home/cuongleduc/buildroot-rv1106/output/build/rv1106-ipc-sdk-e1caff31f64cbb1e7dbed793b764bbf1c020d6c7/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc")
set(CMAKE_CXX_COMPILER "/home/cuongleduc/buildroot-rv1106/output/build/rv1106-ipc-sdk-e1caff31f64cbb1e7dbed793b764bbf1c020d6c7/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++")

# Set sysroot path for cross-compilation
set(CMAKE_SYSROOT "/home/cuongleduc/buildroot-rv1106/output/host/arm-buildroot-linux-uclibcgnueabihf/sysroot")

# Set library search paths
set(CMAKE_LIBRARY_PATH "${CMAKE_SYSROOT}/usr/lib")
set(CMAKE_INCLUDE_PATH "${CMAKE_SYSROOT}/usr/include")

## The /usr/local/Cellar path is by default included
#include_directories(/usr/local/include/opencv4)
#link_directories(/usr/local/lib/)
#target_link_directories(opencv_test PUBLIC /usr/local/Cellar/opencv/4.4.0_1/lib/)

set(OpenCV_DIR "/home/cuongleduc/buildroot-rv1106/output/host/arm-buildroot-linux-uclibcgnueabihf/sysroot/usr/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

add_executable(color_calibration_test src/main.cpp)

# Link libraries with explicit paths to resolve protobuf dependency
target_link_libraries(color_calibration_test 
    opencv_core 
    opencv_mcc 
    opencv_highgui 
    opencv_imgcodecs
    opencv_dnn  # MCC depends on DNN which needs protobuf
    #protobuf    # Explicitly link protobuf
)

# Set RPATH to help with runtime library resolution
#set_target_properties(color_calibration_test PROPERTIES
#    INSTALL_RPATH "${CMAKE_SYSROOT}/usr/lib"
#    BUILD_WITH_INSTALL_RPATH TRUE
#)

#There is no OpenCV cmake files in the path(it's in opencv's source code repo)
#ocv_define_module(opencv_test opencv_core)
