cmake_minimum_required (VERSION 2.8.11)


set(CMAKE_CXX_STANDARD 11)

project (HELLO)

find_library(FFT fftw3)
find_library(PTHREAD pthread)
find_library(X11 X11)

add_executable (fftimg main.cpp)
target_link_libraries(fftimg ${FFT} ${PTHREAD} ${X11})
