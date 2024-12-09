set(EXAMPLE_NAME 19-puma_singularity)
# create an executable
ADD_EXECUTABLE (${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp)

# and link the library against the executable
TARGET_LINK_LIBRARIES (${EXAMPLE_NAME}
	${SAI-PRIMITIVES_LIBRARIES}
	${SAI-PRIMITIVES_EXAMPLES_COMMON_LIBRARIES}
	)

