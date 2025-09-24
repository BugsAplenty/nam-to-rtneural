cmake_minimum_required(VERSION 3.20)
project(NamReamp LANGUAGES CXX)

# ----- Build type -----
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ----- NAM core path (flag or fallback) -----
set(NAMCORE "" CACHE PATH "Path to NeuralAmpModelerCore (root containing NAM/ and Dependencies/)")
if(NAMCORE)
  set(NAMCORE_DIR "${NAMCORE}")
else()
  set(NAMCORE_DIR "${CMAKE_SOURCE_DIR}/deps/NeuralAmpModelerCore")
endif()
message(STATUS "Using NAM Core at: ${NAMCORE_DIR}")
if(NOT EXISTS "${NAMCORE_DIR}/NAM/dsp.h")
  message(FATAL_ERROR
    "NAM core missing.\n"
    "  Expected: ${NAMCORE_DIR}/NAM/dsp.h\n"
    "  Pass -DNAMCORE=/path OR vendor deps/NeuralAmpModelerCore")
endif()

# ----- libsndfile via pkg-config -----
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)  # PkgConfig::SNDFILE

# Prefer system Eigen / nlohmann if available (Nix: pkgs.eigen, pkgs.nlohmann_json)
find_package(Eigen3 QUIET NO_MODULE)        # provides Eigen3::Eigen
find_package(nlohmann_json QUIET)           # provides nlohmann_json::nlohmann_json

if(Eigen3_FOUND)
  message(STATUS "Found system Eigen: ${Eigen3_VERSION}")
else()
  message(STATUS "System Eigen not found; will try NAMCORE/Dependencies/eigen")
endif()

if(nlohmann_json_FOUND)
  message(STATUS "Found system nlohmann_json")
else()
  message(STATUS "System nlohmann_json not found; will try NAMCORE/Dependencies/nlohmann")
endif()

# ----- Sources -----
file(GLOB_RECURSE SRC_CPP "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE NAM_SRC "${NAMCORE_DIR}/NAM/*.cpp")

# >>> create the target BEFORE any target_* calls <<<
add_executable(NamReamp ${SRC_CPP} ${NAM_SRC})

# ----- Includes (put NAM root so "NAM/dsp.h" works) -----
target_include_directories(NamReamp PRIVATE
  "${NAMCORE_DIR}"
)

# Eigen include
if(Eigen3_FOUND)
  target_link_libraries(NamReamp PRIVATE Eigen3::Eigen)
else()
  target_include_directories(NamReamp PRIVATE
    "${NAMCORE_DIR}/Dependencies"
    "${NAMCORE_DIR}/Dependencies/eigen"
  )
endif()

# nlohmann/json: NAM includes "json.hpp" (no prefix), so add the subdir
if(nlohmann_json_FOUND)
  target_link_libraries(NamReamp PRIVATE nlohmann_json::nlohmann_json)
  get_target_property(_NJ_INC nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
  if(_NJ_INC)
    target_include_directories(NamReamp PRIVATE "${_NJ_INC}/nlohmann")
  endif()
else()
  target_include_directories(NamReamp PRIVATE
    "${NAMCORE_DIR}/Dependencies/nlohmann"
  )
endif()

# ----- Link libsndfile -----
target_link_libraries(NamReamp PRIVATE PkgConfig::SNDFILE)

# ----- C++ std & defines -----
target_compile_features(NamReamp PRIVATE cxx_std_17)
target_compile_definitions(NamReamp PRIVATE EIGEN_NO_DEBUG RELEASE=1 NDEBUG=1)

# ----- Optimizations -----
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(NamReamp PRIVATE -ffast-math -fno-rtti -Ofast -pipe -funroll-loops)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(NamReamp PRIVATE -fvectorize)
  else()
    target_compile_options(NamReamp PRIVATE -ftree-vectorize)
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
