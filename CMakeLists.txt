cmake_minimum_required(VERSION 3.20)
project(NamReamp LANGUAGES CXX)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Find libsndfile via pkg-config and create an imported target
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)  # PkgConfig::SNDFILE

# Sources
file(GLOB_RECURSE SRC_CPP
  ${CMAKE_SOURCE_DIR}/src/*.cpp
  ${CMAKE_SOURCE_DIR}/deps/NeuralAmpModelerCore/NAM/*.cpp
)

add_executable(NamReamp ${SRC_CPP})

# Includes (vendor Eigen + nlohmann already in the tree)
target_include_directories(NamReamp PRIVATE
  ${CMAKE_SOURCE_DIR}/deps/NeuralAmpModelerCore
  ${CMAKE_SOURCE_DIR}/deps/NeuralAmpModelerCore/Dependencies
  ${CMAKE_SOURCE_DIR}/deps/NeuralAmpModelerCore/Dependencies/eigen
  ${CMAKE_SOURCE_DIR}/deps/NeuralAmpModelerCore/Dependencies/nlohmann
)

# Link libsndfile cleanly (no manual link_directories / LDFLAGS)
target_link_libraries(NamReamp PRIVATE PkgConfig::SNDFILE)

# C++ standard + per-target options/defs
target_compile_features(NamReamp PRIVATE cxx_std_17)
target_compile_definitions(NamReamp PRIVATE
  EIGEN_NO_DEBUG
  RELEASE=1
  NDEBUG=1
)

# Optimizations (keep them per-target; adjust to taste)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(NamReamp PRIVATE
    -ffast-math -fno-rtti -Ofast -pipe -funroll-loops
  )
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(NamReamp PRIVATE -fvectorize)
  else()
    target_compile_options(NamReamp PRIVATE -ftree-vectorize)
  endif()
endif()

# Make compile_commands.json available (nice for tooling)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
