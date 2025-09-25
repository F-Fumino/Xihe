# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.16)

if (WIN32)
    set(CMAKE_C_COMPILER "clang-cl")
    set(CMAKE_CXX_COMPILER "clang-cl")
else()
    set(CMAKE_C_COMPILER "clang")
    set(CMAKE_CXX_COMPILER "clang++")
    # Linux / macOS 下推荐显式切换到 libc++
    set(CMAKE_CXX_FLAGS "-stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "-lc++abi")
endif()

set(CMAKE_BUILD_TYPE "Release")

if(NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
  include(ProcessorCount)
  ProcessorCount(N)
  
  if(N GREATER 4)
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N} CACHE STRING "Parallel build level" FORCE)
  else()
    set(CMAKE_BUILD_PARALLEL_LEVEL 4 CACHE STRING "Parallel build level" FORCE)
  endif()
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

add_compile_definitions($<$<CONFIG:DEBUG>:XH_DEBUG>)

project ("Xihe")

add_definitions(-DASSETS_DIR=\"${CMAKE_SOURCE_DIR}/assets/\")
add_definitions(-DSHADERS_DIR=\"${CMAKE_SOURCE_DIR}/shaders/\")
add_definitions(-DOUTPUT_DIR=\"${CMAKE_SOURCE_DIR}/output/\")

add_subdirectory("third_party")

set(SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
# Include sub-projects.
add_subdirectory ("xihe")