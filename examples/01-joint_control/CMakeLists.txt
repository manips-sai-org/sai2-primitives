# create an executable
ADD_EXECUTABLE (01-joint_control 01-joint_control.cpp)

# and link the library against the executable
TARGET_LINK_LIBRARIES (01-joint_control
	${SAI2-PRIMITIVES_LIBRARIES}
	${SAI2-PRIMITIVES_EXAMPLES_COMMON_LIBRARIES}
	)

# export resources such as model files.
# NOTE: this requires an install build
SET(RESOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/resources)
FILE(MAKE_DIRECTORY ${RESOURCE_DIR})
FILE(COPY world.urdf puma.urdf DESTINATION ${RESOURCE_DIR})
