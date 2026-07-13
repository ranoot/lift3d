cmake_minimum_required(VERSION 3.22)
project(SampleCppForInterns LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Assume we have system Eigen and Opencv
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

# Static library for GMDBase Structs
add_library(GMDBase 
    STATIC
    GMDBase/Entity/TimeStamp.cpp # add other cpp files here
)
target_include_directories(GMDBase PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(CommonLibrary
    STATIC
    Common/Entity/Transformation.cpp
    Common/Algorithm/ImageUtils.cpp
    Common/Algorithm/PoseUtils.cpp
)
target_include_directories(CommonLibrary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(CommonLibrary PRIVATE Eigen3::Eigen ${OpenCV_LIBS})

add_executable(sampleMain
    main.cpp
)
target_link_libraries(sampleMain GMDBase CommonLibrary Eigen3::Eigen ${OpenCV_LIBS})
