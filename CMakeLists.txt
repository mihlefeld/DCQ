cmake_minimum_required(VERSION 3.18)
project(DCQ)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(DCQ
        main.cpp dcq/algorithm.h dcq/init.h dcq/types.h dcq/utils.h
        dcq/algorithm.cpp dcq/init.cpp dcq/utils.cpp
        dcq/PBar.cpp dcq/PBar.h)

target_link_libraries(DCQ ${OpenCV_LIBS})
target_link_libraries(DCQ ${TORCH_LIBRARIES})
