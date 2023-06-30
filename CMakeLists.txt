cmake_minimum_required(VERSION 3.18)
project(DCQ)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(DCQ main.cpp dcq/algorithm.h dcq/init.h dcq/types.h dcq/utils.h
        dcq/algorithm.cpp dcq/init.cpp dcq/utils.cpp
        )

#string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
#string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -g")
#string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -Os -DNDEBUG")
#string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -O3 -DNDEBUG")
#string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -O2 -g -DNDEBUG")

add_compile_options(-O3)

target_link_libraries(DCQ ${OpenCV_LIBS})
target_link_libraries(DCQ ${TORCH_LIBRARIES})
#set_property(TARGET DCQ PROPERTY CXX_STANDARD 17)
