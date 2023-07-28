CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

PROJECT (SAI2-PRIMITIVES)
SET (PROJECT_VERSION 0.1.0) #major version is zero for now since api will certainly change in future

SET(CMAKE_CXX_FLAGS "-std=c++11 -I/usr/include -I/usr/local/include")
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  MESSAGE(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  # Set the default build, and also the possible values of build type for cmake-gui
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()

# option to use OTG 
option(USE_OTG "use the Reflexxes library for trajectory generation" OFF)
if(USE_OTG)
	# need to set these two paths manually depending on install path of Reflexxes library
	SET (OTG_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../external/ReflexxesTypeII) # need to manually set the path

	if (CMAKE_SYSTEM_NAME MATCHES Darwin) 
		SET (OTG_LIBRARY ${OTG_SOURCE_DIR}/MacOS/x64/release/lib/libReflexxesTypeII.a)
	elseif (CMAKE_SYSTEM_NAME MATCHES Linux)
		SET (OTG_LIBRARY ${OTG_SOURCE_DIR}/Linux/x64/release/lib/shared/libReflexxesTypeII.so)
	endif () #NOTE: Windows is currently unsupported

	SET (PROJECT_DEFINITIONS -DUSING_OTG)
	SET (OTG_INCLUDE_DIRS ${OTG_SOURCE_DIR}/include)
	SET (OTG_SOURCE 
		${PROJECT_SOURCE_DIR}/src/trajectory_generation/OTG.cpp
		${PROJECT_SOURCE_DIR}/src/trajectory_generation/OTG_ori.cpp
		${PROJECT_SOURCE_DIR}/src/trajectory_generation/OTG_posori.cpp
		)
	INCLUDE_DIRECTORIES(${OTG_INCLUDE_DIRS})
	add_definitions(-DUSING_OTG)
endif()

get_directory_property( DirDefs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS )
foreach( d ${DirDefs} )
    message( STATUS "Found Define: " ${d} )
endforeach()
message( STATUS "DirDefs: " ${DirDefs} )

# include Eigen
set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
if(NOT EIGEN3_INCLUDE_DIR)
	find_package(Eigen3 QUIET)
	# TODO: Find automatic way to find for Ubuntu 14.04
	if(NOT EIGEN3_INCLUDE_DIR)
		set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)
	endif()
endif()

# include sai2-model
find_package(SAI2-MODEL REQUIRED)

# add tasks
SET(TASKS_SOURCE 
	# ${PROJECT_SOURCE_DIR}/src/tasks/PositionTask.cpp
	# ${PROJECT_SOURCE_DIR}/src/tasks/OrientationTask.cpp
	${PROJECT_SOURCE_DIR}/src/tasks/PosOriTask.cpp
	${PROJECT_SOURCE_DIR}/src/tasks/JointTask.cpp
	# ${PROJECT_SOURCE_DIR}/src/tasks/TwoHandTwoRobotsTask.cpp
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
	)

# Create the library
if(USE_OTG)
	ADD_LIBRARY (sai2-primitives STATIC
		${TASKS_SOURCE}
		# ${HAPTIC_TASKS_SOURCE}
		${OTG_SOURCE}
		)
else()
	ADD_LIBRARY (sai2-primitives STATIC
		${TASKS_SOURCE}
		# ${HAPTIC_TASKS_SOURCE}
		)
endif()


SET(SAI2-PRIMITIVES_LIBRARIES
	sai2-primitives
	)

if(USE_OTG)
	SET(SAI2-PRIMITIVES_LIBRARIES
	sai2-primitives
	${OTG_LIBRARY}
	${SAI2-MODEL_LIBRARIES}
	)
else()
	SET(SAI2-PRIMITIVES_LIBRARIES
	sai2-primitives
	${SAI2-MODEL_LIBRARIES}
	)
endif()


SET (SAI2-PRIMITIVES_DEFINITIONS ${PROJECT_DEFINITIONS})

# 
# export package
# 

# export package for use from the build tree
EXPORT (TARGETS sai2-primitives FILE ${PROJECT_BINARY_DIR}/SAI2-PRIMITIVESTargets.cmake)
EXPORT (PACKAGE SAI2-PRIMITIVES)

# configure package
SET (CONF_INCLUDE_DIRS ${SAI2-PRIMITIVES_INCLUDE_DIRS} ${OTG_INCLUDE_DIRS})
SET (CONF_LIBRARIES ${SAI2-PRIMITIVES_LIBRARIES})
SET (CONF_DEFINITIONS ${SAI2-PRIMITIVES_DEFINITIONS})

# message(Project definitions : \n ${SAI2-PRIMITIVES_DEFINITIONS} \n\n)

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
