cmake_minimum_required(VERSION 3.5)
project( PUPIL )
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Glog REQUIRED)
find_package(GFlags REQUIRED)
find_package(OpenCV REQUIRED)


set(PUPIL_LIBRARIES
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${OpenCV_LIBS})

include_directories( ${PROJECT_SOURCE_DIR}/src )
include_directories(${PROJECT_SOURCE_DIR}/third_party)

add_subdirectory(src)