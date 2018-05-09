# create an executable
ADD_EXECUTABLE (05-kuka_pick_and_place 05-kuka_pick_and_place.cpp)
ADD_EXECUTABLE (marker marker.cpp)

# and link the library against the executable
TARGET_LINK_LIBRARIES (05-kuka_pick_and_place
	${SAI2-PRIMITIVES_LIBRARIES}
	${SAI2-PRIMITIVES_EXAMPLES_COMMON_LIBRARIES}
	)
TARGET_LINK_LIBRARIES(marker ${OpenCV_LIBS})

# export resources such as model files.
# NOTE: this requires an install build
SET(RESOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/resources)
FILE(MAKE_DIRECTORY ${RESOURCE_DIR})
FILE(COPY world.urdf kuka_iiwa.urdf cube.urdf DESTINATION ${RESOURCE_DIR})
