
add_library(common fits_helper.cpp)
target_include_directories(common PUBLIC
	include
	${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(common PUBLIC ${OpenCV_LIBS})