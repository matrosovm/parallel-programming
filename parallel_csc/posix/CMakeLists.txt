cmake_minimum_required(VERSION 3.17)

project(csc_2021_01_posix_matrosovm)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic -pthread")

add_executable(csc_2021_01_posix_matrosovm src/main.cpp)