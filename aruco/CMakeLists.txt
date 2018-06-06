# set(the_description "ArUco Marker Detection")
# ocv_define_module(aruco opencv_core opencv_imgproc opencv_calib3d WRAP python java)

# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable(create_board_charuco samples/create_board_charuco.cpp)
add_executable(calibrate_camera_charuco samples/calibrate_camera_charuco.cpp)
add_executable(detect_markers samples/detect_markers.cpp)

# Link your application with OpenCV libraries
target_link_libraries(create_board_charuco ${OpenCV_LIBS})
target_link_libraries(calibrate_camera_charuco ${OpenCV_LIBS})
target_link_libraries(detect_markers ${OpenCV_LIBS})