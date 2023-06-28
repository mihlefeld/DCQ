cmake_minimum_required(VERSION 3.18)
project(DCQ)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(DCQ main.cpp utils/utils.h utils/utils.cpp)

target_link_libraries(DCQ ${OpenCV_LIBS})
target_link_libraries(DCQ ${TORCH_LIBRARIES})
#set_property(TARGET DCQ PROPERTY CXX_STANDARD 17)
