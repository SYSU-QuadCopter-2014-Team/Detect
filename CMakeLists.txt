cmake_minimum_required(VERSION 2.8)
project(test)

find_package(OpenCV REQUIRED)
if( OpenCV_FOUND )
list( APPEND ThirdParty_LIBS ${OpenCV_LIBS} )
    include_directories( ${OpenCV_INCLUDE_DIRS} )
endif( OpenCV_FOUND )

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src) 
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")
add_executable( RunTracker ${sourcefiles} )
target_link_libraries( RunTracker ${OpenCV_LIBS})




