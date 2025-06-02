cmake_minimum_required(VERSION 3.15)

project(AuroraGCAFS VERSION 1.0.0 LANGUAGES Fortran C CXX)

# Add the cmake directory to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Option for installation path
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Set default install paths if not provided
if(NOT DEFINED CMAKE_INSTALL_BINDIR)
  set(CMAKE_INSTALL_BINDIR bin)
endif()

# Add the source code subdirectories
add_subdirectory(sorc/auroragcafs)
