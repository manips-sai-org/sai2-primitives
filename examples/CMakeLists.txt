# include Chai3d
find_package(CHAI3D REQUIRED)
include_directories(${CHAI3D_INCLUDE_DIRS})
add_definitions(${CHAI3D_DEFINITIONS})
link_directories(${CHAI3D_LIBRARY_DIRS})

# SAI-SIMULATION
find_package(SAI-SIMULATION REQUIRED)
include_directories(${SAI-SIMULATION_INCLUDE_DIRS})
add_definitions(${SAI-SIMULATION_DEFINITIONS})

# SAI-GRAPHICS
find_package(SAI-GRAPHICS REQUIRED)
include_directories(${SAI-GRAPHICS_INCLUDE_DIRS})
add_definitions(${SAI-GRAPHICS_DEFINITIONS})

# SAI-COMMON
find_package(SAI-COMMON REQUIRED)
include_directories(${SAI-COMMON_INCLUDE_DIRS})

# SAI-URDF
find_package(SAI-URDF REQUIRED)
include_directories(${SAI-URDF_INCLUDE_DIRS})

set(SAI-PRIMITIVES_EXAMPLES_COMMON_LIBRARIES
${SAI-MODEL_LIBRARIES}
${SAI-GRAPHICS_LIBRARIES}
${SAI-SIMULATION_LIBRARIES}
${SAI-COMMON_LIBRARIES}
${SAI-URDF_LIBRARIES}
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
add_subdirectory(18-panda_singularity)
add_subdirectory(19-puma_singularity)
