# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.16)

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