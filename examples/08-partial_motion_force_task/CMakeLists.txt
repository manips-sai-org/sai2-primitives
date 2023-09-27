set(EXAMPLE_NAME 08-partial_motion_force_task)
# create an executable
ADD_EXECUTABLE (${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp)

# and link the library against the executable
TARGET_LINK_LIBRARIES (${EXAMPLE_NAME}
	${SAI2-PRIMITIVES_LIBRARIES}
	${SAI2-PRIMITIVES_EXAMPLES_COMMON_LIBRARIES}
	)

# export resources such as model files.
SET(RESOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/resources)
FILE(MAKE_DIRECTORY ${RESOURCE_DIR})
FILE(COPY world.urdf puma.urdf DESTINATION ${RESOURCE_DIR})
