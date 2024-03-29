set(EXAMPLE_NAME 17-bilateral_teleop_with_POPC)
# create an executable
add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp)

# and link the library against the executable
target_link_libraries(${EXAMPLE_NAME} ${SAI2-PRIMITIVES_LIBRARIES}
                      ${SAI2-PRIMITIVES_EXAMPLES_COMMON_LIBRARIES})
