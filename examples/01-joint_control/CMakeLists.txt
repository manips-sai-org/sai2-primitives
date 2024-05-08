set(EXAMPLE_NAME 01-joint_control)

# create an executable
ADD_EXECUTABLE (${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp)

# and link the library against the executable
TARGET_LINK_LIBRARIES (${EXAMPLE_NAME}
	${SAI2-PRIMITIVES_LIBRARIES}
	${SAI2-PRIMITIVES_EXAMPLES_COMMON_LIBRARIES}
	)