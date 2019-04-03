cmake_minimum_required(VERSION 2.8)
  
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -pipe")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O2 -pipe")

project(lanczos3)

add_executable(lanczos3
    main.cpp

    gpu.c
    cpu.c

    opencl.c
    kernel.c
    timer.c
)

target_link_libraries(lanczos3
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_highgui

    OpenCL
)
