set(EXAMPLE_NAME 15-haptic_control_impedance_type)
# create an executable
add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp)

# and link the library against the executable
target_link_libraries(${EXAMPLE_NAME} ${SAI2-PRIMITIVES_LIBRARIES} ${SAI2-PRIMITIVES_EXAMPLES_COMMON_LIBRARIES})

# export resources such as model files.
set(RESOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/resources)
file(MAKE_DIRECTORY ${RESOURCE_DIR})
file(COPY world.urdf panda_arm.urdf DESTINATION ${RESOURCE_DIR})
