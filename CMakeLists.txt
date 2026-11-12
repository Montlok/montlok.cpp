# montlok as a C/C++ library (libmontlok), for C++ clients such as the Qt
# terminal. setup.py builds the Python package and bindings/rust the Rust crate
# from the same sources.
#
#   cmake -S . -B build && cmake --build build && ctest --test-dir build
#
# Consumers: add_subdirectory / FetchContent, then
#   target_link_libraries(app PRIVATE montlok::montlok)   # #include <montlok/montlok.hpp>
cmake_minimum_required(VERSION 3.21)
project(montlok VERSION 0.2.0 LANGUAGES C CXX)

option(MONTLOK_BUILD_SHARED "Build libmontlok as a shared library" OFF)
option(MONTLOK_BUILD_TESTS "Build the C++ tests" ${PROJECT_IS_TOP_LEVEL})
option(MONTLOK_BUILD_EXAMPLES "Build the example plugin" ${PROJECT_IS_TOP_LEVEL})
option(MONTLOK_BUILD_TOOLS "Build montlok_bench" ${PROJECT_IS_TOP_LEVEL})

find_package(Threads REQUIRED)

set(MONTLOK_SOURCES
    src/core/archive.cpp
    src/core/c_api.cpp
    src/core/json.cpp
    src/core/plugin.cpp
    src/core/runtime.cpp
    src/core/sequential.cpp
    src/core/trees.cpp
    microstructure/bindings.cpp)

if(MONTLOK_BUILD_SHARED)
  add_library(montlok SHARED ${MONTLOK_SOURCES})
  target_compile_definitions(montlok PUBLIC MONTLOK_SHARED PRIVATE MONTLOK_BUILDING)
else()
  add_library(montlok STATIC ${MONTLOK_SOURCES})
endif()
add_library(montlok::montlok ALIAS montlok)

target_compile_features(montlok PUBLIC cxx_std_17)
set_target_properties(montlok PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON
  VERSION ${PROJECT_VERSION})
target_include_directories(montlok
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE src/core microstructure)
if(NOT MSVC)
  # The flags of the Python and Rust builds.
  target_compile_options(montlok PRIVATE -fno-math-errno -ffp-contract=fast $<$<CONFIG:Release,RelWithDebInfo>:-O3>)
endif()
target_link_libraries(montlok PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS montlok EXPORT montlokTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/montlok DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT montlokTargets NAMESPACE montlok:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/montlok)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/montlokConfig.cmake
  "include(CMakeFindDependencyMacro)\nfind_dependency(Threads)\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/montlokTargets.cmake\")\n")
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/montlokConfigVersion.cmake COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/montlokConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/montlokConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/montlok)

if(MONTLOK_BUILD_EXAMPLES)
  add_library(ema_dot MODULE examples/plugin/ema_dot.c)
  target_include_directories(ema_dot PRIVATE include)
  set_target_properties(ema_dot PROPERTIES C_STANDARD 11 PREFIX "lib")
endif()

if(MONTLOK_BUILD_TOOLS)
  add_executable(montlok_bench tools/bench_latency.cpp)
  target_link_libraries(montlok_bench PRIVATE montlok::montlok)
endif()

if(MONTLOK_BUILD_TESTS)
  enable_testing()
  add_executable(montlok_test_core tests/cpp/test_core.cpp)
  target_link_libraries(montlok_test_core PRIVATE montlok::montlok)
  add_test(NAME core COMMAND montlok_test_core ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
  if(MONTLOK_BUILD_EXAMPLES)
    target_compile_definitions(montlok_test_core PRIVATE MONTLOK_TEST_PLUGIN="$<TARGET_FILE:ema_dot>")
    add_dependencies(montlok_test_core ema_dot)
  endif()
endif()
