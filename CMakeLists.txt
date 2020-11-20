cmake_minimum_required(VERSION 3.5)
project( PUPIL )
set (CMAKE_CXX_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR}/third_party)
find_package(OpenCV REQUIRED PATHS "C:/OpenCV-2.4.9/opencv/build/x64/vc12/lib")

set(PUPIL_LIBRARIES ${OpenCV_LIBS})

include_directories( "C:/OpenCV-2.4.9/opencv/build/include")
include_directories( "C:/OpenCV-2.4.9/opencv/build/include/opencv")
include_directories( "C:/OpenCV-2.4.9/opencv/build/include/opencv2")
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory(src)