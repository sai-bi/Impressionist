cmake_minimum_required(VERSION 2.8)

project( impressionist )
find_package( OpenCV REQUIRED )

SET(SRC_LIST impressionist.cpp impressionist.h test.cpp)
add_executable(test ${SRC_LIST})
target_link_libraries(test ${OpenCV_LIBS} )
