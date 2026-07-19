
include_directories(${OpenCV_INCLUDE_DIRS})

add_cuda_hip_library(common fits_helper.cu)
target_sources(common PRIVATE fits_helper.cpp)

target_include_directories(common PUBLIC
	include
	${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(common PUBLIC ${OpenCV_LIBS})