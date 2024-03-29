# include Chai3d
find_package(CHAI3D REQUIRED)
include_directories(${CHAI3D_INCLUDE_DIRS})
add_definitions(${CHAI3D_DEFINITIONS})
link_directories(${CHAI3D_LIBRARY_DIRS})

# SAI2-SIMULATION
find_package(SAI2-SIMULATION REQUIRED)
include_directories(${SAI2-SIMULATION_INCLUDE_DIRS})
add_definitions(${SAI2-SIMULATION_DEFINITIONS})

# SAI2-GRAPHICS
find_package(SAI2-GRAPHICS REQUIRED)
include_directories(${SAI2-GRAPHICS_INCLUDE_DIRS})
add_definitions(${SAI2-GRAPHICS_DEFINITIONS})

# SAI2-COMMON
find_package(SAI2-COMMON REQUIRED)
include_directories(${SAI2-COMMON_INCLUDE_DIRS})

# SAI2-URDF
find_package(SAI2-URDF REQUIRED)
include_directories(${SAI2-URDF_INCLUDE_DIRS})

set(SAI2-PRIMITIVES_EXAMPLES_COMMON_LIBRARIES
${SAI2-MODEL_LIBRARIES}
${SAI2-GRAPHICS_LIBRARIES}
${SAI2-SIMULATION_LIBRARIES}
${SAI2-COMMON_LIBRARIES}
${SAI2-URDF_LIBRARIES}
${CHAI3D_LIBRARIES}
)

# examples folder
set(EXAMPLES_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DEXAMPLES_FOLDER="${EXAMPLES_FOLDER}")

add_subdirectory(01-joint_control)
add_subdirectory(02-joint_control_internal_otg)
add_subdirectory(03-cartesian_motion_control)
add_subdirectory(04-task_and_redundancy)
add_subdirectory(05-using_robot_controller)
add_subdirectory(06-partial_joint_task)
add_subdirectory(07-surface_surface_contact)
add_subdirectory(08-partial_motion_force_task)
add_subdirectory(09-3d_position_force_controller)
add_subdirectory(10-3d_orientation_controller)
add_subdirectory(11-planar_robot_controller)
add_subdirectory(15-haptic_control_impedance_type)
add_subdirectory(16-haptic_control_admittance_type)
add_subdirectory(17-bilateral_teleop_with_POPC)
