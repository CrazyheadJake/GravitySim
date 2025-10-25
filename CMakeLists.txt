cmake_minimum_required(VERSION 3.16.0)


find_package(CUDAToolkit REQUIRED)  # Already finds headers/libs

# For NVCC compiler
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options /bigobj")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

project(GravitySim LANGUAGES CXX CUDA)

# we define MY_SOURCES to be a list of all the source files for my project
file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

set(CMAKE_CUDA_ARCHITECTURES "native")
# Add your sources
add_executable(${PROJECT_NAME}
    ${MY_SOURCES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)

# Tell CMake where to find CUDA headers for both .cu and .cpp files
target_include_directories(${PROJECT_NAME}
    PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    ${CUDAToolkit_INCLUDE_DIRS}      # <<< this line is key
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)

set(RAYLIB_LIB "${CMAKE_SOURCE_DIR}/libs/libraylib_msvc.lib")
# Link the CUDA libraries (e.g., cudart) — needed for host calls like cudaMalloc
target_link_libraries(${PROJECT_NAME}
    PUBLIC
    CUDA::cudart                    # safer than hardcoded paths
    ${RAYLIB_LIB}
    winmm
    gdi32
    opengl32
    shell32
)


# target_compile_options(${PROJECT_NAME} PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:
#         --diag-suppress=177   # Suppress warnings
#         # --ptxas-options=-v
#         # --maxrregcount=0
#         # -O0
#     >
#     $<$<COMPILE_LANGUAGE:CXX>:
#         /Od
#     >
# )

message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "BUILD TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CUDA Release flags: ${CMAKE_CUDA_FLAGS_RELEASE}")