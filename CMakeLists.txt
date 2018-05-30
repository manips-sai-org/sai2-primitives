CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

PROJECT (SAI2-PRIMITIVES)
SET (PROJECT_VERSION 0.1.0) #major version is zero for now since api will certainly change in future

SET(CMAKE_CXX_FLAGS "-std=c++11 -I/usr/include -I/usr/local/include")
SET(CMAKE_BUILD_TYPE "Debug")

# include Eigen
set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
if(NOT EIGEN3_INCLUDE_DIR)
	find_package(Eigen3 QUIET)
	# TODO: Find automatic way to find for Ubuntu 14.04
	if(NOT EIGEN3_INCLUDE_DIR)
		set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)
	endif()
endif()

# include opencv
find_package(OpenCV REQUIRED)

# include sai2-model
find_package(SAI2-MODEL REQUIRED)

# add tasks
SET(TASKS_SOURCE 
	${PROJECT_SOURCE_DIR}/src/tasks/PositionTask.cpp
	${PROJECT_SOURCE_DIR}/src/tasks/OrientationTask.cpp
	${PROJECT_SOURCE_DIR}/src/tasks/PosOriTask.cpp
	${PROJECT_SOURCE_DIR}/src/tasks/JointTask.cpp
	)

# add primitives 
SET(PRIMITIVES_SOURCE 
	${PROJECT_SOURCE_DIR}/src/primitives/RedundantArmMotion.cpp
	${PROJECT_SOURCE_DIR}/src/primitives/SurfaceSurfaceAlignment.cpp
	${PROJECT_SOURCE_DIR}/src/primitives/PickAndPlace.cpp
	${PROJECT_SOURCE_DIR}/src/primitives/AllegroGrasp.cpp
	)

# add header files
SET(SAI2-PRIMITIVES_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)


# - SAI2-COMMON
FIND_PACKAGE(SAI2-COMMON REQUIRED)
include_directories(${SAI2-COMMON_INCLUDE_DIRS})
add_definitions(${SAI2-COMMON_DEFINITIONS})

# Add the include directory to the include paths
INCLUDE_DIRECTORIES (
	${SAI2-PRIMITIVES_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${SAI2-MODEL_INCLUDE_DIRS}
	${SAI2-COMMON_LIBRARIES}
	)

# Create the library
ADD_LIBRARY (sai2-primitives STATIC
	${TASKS_SOURCE}
	${PRIMITIVES_SOURCE}
	)

SET(SAI2-PRIMITIVES_LIBRARIES
	sai2-primitives
	)

# 
# export package
# 

# export package for use from the build tree
EXPORT (TARGETS sai2-primitives FILE ${PROJECT_BINARY_DIR}/SAI2-PRIMITIVESTargets.cmake)
EXPORT (PACKAGE SAI2-PRIMITIVES)

# configure package
SET (CONF_INCLUDE_DIRS ${SAI2-PRIMITIVES_INCLUDE_DIRS} ${RBDL_INCLUDE_DIR})
SET (CONF_LIBRARIES ${SAI2-PRIMITIVES_LIBRARIES})

# create package configuration and version files
CONFIGURE_FILE(SAI2-PRIMITIVESConfig.cmake.in "${PROJECT_BINARY_DIR}/SAI2-PRIMITIVESConfig.cmake" @ONLY)
INCLUDE(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/SAI2-PRIMITIVESConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# add examples
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/examples)
