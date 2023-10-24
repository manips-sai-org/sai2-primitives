CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT (SAI2-PRIMITIVES)
SET (PROJECT_VERSION 0.1.0) #major version is zero for now since api will certainly change in future

SET(CMAKE_CXX_FLAGS "-std=c++17 -I/usr/include -I/usr/local/include -fPIC")
if (${CMAKE_SYSTEM_NAME} MATCHES Darwin)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/include")
endif ()

IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  MESSAGE(STATUS "Setting build type to 'Release' as none was specified.")
  # Set the default build, and also the possible values of build type for cmake-gui
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()

# include Eigen
find_package(Eigen3 REQUIRED)

# include sai2-model
find_package(SAI2-MODEL REQUIRED)

# include ruckig for OTG
set(RUCKIG_LOCAL_DIR ${PROJECT_SOURCE_DIR}/ruckig)
set(RUCKIG_INCLUDE_DIR ${RUCKIG_LOCAL_DIR}/include/)
FIND_LIBRARY(RUCKIG_LIBRARIES ruckig PATHS ${RUCKIG_LOCAL_DIR}/build NO_DEFAULT_PATH)

# add tasks
SET(TASKS_SOURCE 
	${PROJECT_SOURCE_DIR}/src/RobotController.cpp
	${PROJECT_SOURCE_DIR}/src/tasks/MotionForceTask.cpp
	${PROJECT_SOURCE_DIR}/src/tasks/JointTask.cpp
	# ${PROJECT_SOURCE_DIR}/src/tasks/TwoHandTwoRobotsTask.cpp
	)

# add helper modules
SET(HELPER_MODULES_SOURCE 
	${PROJECT_SOURCE_DIR}/src/helper_modules/POPCExplicitForceControl.cpp
	${PROJECT_SOURCE_DIR}/src/helper_modules/OTG_joints.cpp
	${PROJECT_SOURCE_DIR}/src/helper_modules/OTG_6dof_cartesian.cpp
	)

# # add haptic_tasks
# SET(HAPTIC_TASKS_SOURCE
# 	${PROJECT_SOURCE_DIR}/src/haptic_tasks/OpenLoopTeleop.cpp
# 	${PROJECT_SOURCE_DIR}/src/haptic_tasks/HapticController.cpp
# 	${PROJECT_SOURCE_DIR}/src/haptic_tasks/BilateralPassivityController.cpp
# 	${PROJECT_SOURCE_DIR}/src/haptic_tasks/ImpedanceControl.cpp
# 	${PROJECT_SOURCE_DIR}/src/haptic_tasks/PositionControl.cpp
# 	)

# add header files
SET(SAI2-PRIMITIVES_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)

# Add the include directory to the include paths
INCLUDE_DIRECTORIES (
	${SAI2-PRIMITIVES_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${SAI2-MODEL_INCLUDE_DIRS}
	${RUCKIG_INCLUDE_DIR}
	)

# Create the library
ADD_LIBRARY (sai2-primitives STATIC
	${TASKS_SOURCE}
	${HELPER_MODULES_SOURCE}
	# ${HAPTIC_TASKS_SOURCE}
	)

SET(SAI2-PRIMITIVES_LIBRARIES
	sai2-primitives
	# ${SAI2-MODEL_LIBRARIES}
	${RUCKIG_LIBRARIES}
	)

SET (SAI2-PRIMITIVES_DEFINITIONS ${PROJECT_DEFINITIONS})

# 
# export package
# 

# export package for use from the build tree
EXPORT (TARGETS sai2-primitives FILE ${PROJECT_BINARY_DIR}/SAI2-PRIMITIVESTargets.cmake)
EXPORT (PACKAGE SAI2-PRIMITIVES)

# configure package
SET (CONF_INCLUDE_DIRS ${SAI2-PRIMITIVES_INCLUDE_DIRS} ${RUCKIG_INCLUDE_DIRS})
SET (CONF_LIBRARIES ${SAI2-PRIMITIVES_LIBRARIES})
SET (CONF_DEFINITIONS ${SAI2-PRIMITIVES_DEFINITIONS})

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
