cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(graphFusion
    VERSION 1.0.1
    LANGUAGES CXX
)

if(BUTTERSCOTCH_COMPILE_CUDA)
  enable_language(CUDA)
  list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
endif()

find_package(SEAL 4.1 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

if(NOT Python3_VERSION_MAJOR EQUAL 3)
    message(FATAL_ERROR "EVA requires Python 3. Please ensure you have it
        installed in a location searched by CMake.")
endif()

# add_library(fusionFHE STATIC
#     fusion.cpp
#     codegen.cpp
# )

add_executable(graphFusion GraphFusion.cpp test.cpp)

target_include_directories(graphFusion
    PRIVATE
    ${PROJECT_SOURCE_DIR}/ButterscotchFHE-Viewing/butterscotchFHE
    # Add other include directories as needed
)