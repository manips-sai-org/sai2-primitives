cmake_minimum_required(VERSION 3.10)

project(SAI2-PRIMITIVES)
set(PROJECT_VERSION 0.1.0)

set(CMAKE_CXX_FLAGS "-std=c++17 -I/usr/include -I/usr/local/include -fPIC")
if(${CMAKE_SYSTEM_NAME} MATCHES Darwin)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/include")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")

  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

# include Eigen
find_package(Eigen3 REQUIRED)

# include sai2-model
find_package(SAI2-MODEL REQUIRED)

# include ruckig for OTG
set(RUCKIG_LOCAL_DIR ${PROJECT_SOURCE_DIR}/ruckig)
set(RUCKIG_INCLUDE_DIR ${RUCKIG_LOCAL_DIR}/include/)
find_library(
  RUCKIG_LIBRARIES ruckig
  PATHS ${RUCKIG_LOCAL_DIR}/build
  NO_DEFAULT_PATH)

# add tasks
set(CONTROLLERS_SOURCE
    ${PROJECT_SOURCE_DIR}/src/RobotController.cpp
    ${PROJECT_SOURCE_DIR}/src/tasks/MotionForceTask.cpp
    ${PROJECT_SOURCE_DIR}/src/tasks/JointTask.cpp
    ${PROJECT_SOURCE_DIR}/src/HapticDeviceController.cpp
    ${PROJECT_SOURCE_DIR}/src/POPCBilateralTeleoperation.cpp)

# add helper modules
set(HELPER_MODULES_SOURCE
    ${PROJECT_SOURCE_DIR}/src/helper_modules/POPCExplicitForceControl.cpp
    ${PROJECT_SOURCE_DIR}/src/helper_modules/OTG_joints.cpp
    ${PROJECT_SOURCE_DIR}/src/helper_modules/OTG_6dof_cartesian.cpp
    ${PROJECT_SOURCE_DIR}/src/helper_modules/Sai2PrimitivesCommonDefinitions.cpp)

# add header files
set(SAI2-PRIMITIVES_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src
                                 ${RUCKIG_LOCAL_DIR}/include/)

# Add the include directory to the include paths
include_directories(
	${SAI2-PRIMITIVES_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${SAI2-MODEL_INCLUDE_DIRS}
	${RUCKIG_INCLUDE_DIR})

# Create the library
add_library(sai2-primitives STATIC ${CONTROLLERS_SOURCE}
                                   ${HELPER_MODULES_SOURCE})

set(SAI2-PRIMITIVES_LIBRARIES sai2-primitives ${RUCKIG_LIBRARIES})

set(SAI2-PRIMITIVES_DEFINITIONS ${PROJECT_DEFINITIONS})

#
# export package
#

# export package for use from the build tree
export(TARGETS sai2-primitives
       FILE ${PROJECT_BINARY_DIR}/SAI2-PRIMITIVESTargets.cmake)
export(PACKAGE SAI2-PRIMITIVES)

# configure package
set(CONF_INCLUDE_DIRS ${SAI2-PRIMITIVES_INCLUDE_DIRS})
set(CONF_LIBRARIES ${SAI2-PRIMITIVES_LIBRARIES})
set(CONF_DEFINITIONS ${SAI2-PRIMITIVES_DEFINITIONS})

# create package configuration and version files
configure_file(SAI2-PRIMITIVESConfig.cmake.in
               "${PROJECT_BINARY_DIR}/SAI2-PRIMITIVESConfig.cmake" @ONLY)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/SAI2-PRIMITIVESConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

# add examples
add_subdirectory(${PROJECT_SOURCE_DIR}/examples)
