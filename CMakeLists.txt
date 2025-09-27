project(cuda_stacker)

cmake_minimum_required(VERSION 4.0)

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit found: ${CUDAToolkit_VERSION}")
    if (CPU_ONLY)
        message(STATUS "Compiling only CPU version")
    elseif (ROCM)
        find_package(HIP REQUIRED)
        enable_language(HIP)
        set(HIP_ARCHITECTURES "gfx1030")
        message(STATUS "Compiling with ROCM support")
    else()
        enable_language(CUDA)
        message(STATUS "Compiling with CUDA support")
    endif()
else()
    message(STATUS "CUDAToolkit not found. Comiling only CPU version")
endif()


function(add_cuda_hip_executable target_name cuda_source)
    if(ROCM)
        # Hipify the CUDA file
        set(HIPIFIED_FILE ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.hip)
        add_custom_command(
            OUTPUT ${HIPIFIED_FILE}
            COMMAND hipify-clang ${CMAKE_CURRENT_SOURCE_DIR}/${cuda_source} -o ${HIPIFIED_FILE} --cuda-path ${CUDAToolkit_LIBRARY_ROOT} 
                    -I${CMAKE_SOURCE_DIR}/src 
                    -I${CMAKE_SOURCE_DIR}/src/common 
                    -I${CMAKE_SOURCE_DIR}/src/debayer 
                    -I${CMAKE_SOURCE_DIR}/src/debayer/include
                    -I${CMAKE_SOURCE_DIR}/src/stacker 
                    -I${CMAKE_SOURCE_DIR}/src/star_finder
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${cuda_source}
            COMMENT "Hipifying ${cuda_source}"
        )
        add_executable(${target_name} ${HIPIFIED_FILE})

        # Add include directories for the compiler to find project headers
        # CMAKE_CURRENT_SOURCE_DIR refers to the directory of the CMakeLists.txt that *calls* this function
        target_include_directories(${target_name} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}  # For local headers like device_alfa_sigma.h
            ${CMAKE_SOURCE_DIR}/src      # For common headers like ../common/fits_api.h
        )
    else()
        add_executable(${target_name} ${cuda_source})
    endif()
endfunction()

function(add_cuda_hip_library target_name cuda_source)
    if(ROCM)
        # Hipify the CUDA file
        set(HIPIFIED_FILE ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.hip)
        add_custom_command(
            OUTPUT ${HIPIFIED_FILE}
            COMMAND hipify-clang ${CMAKE_CURRENT_SOURCE_DIR}/${cuda_source} -o ${HIPIFIED_FILE} --cuda-path ${CUDAToolkit_LIBRARY_ROOT} -I${CMAKE_SOURCE_DIR}/src
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${cuda_source}
            COMMENT "Hipifying ${cuda_source}"
        )
        add_library(${target_name} ${HIPIFIED_FILE})
    else()
        add_library(${target_name} ${cuda_source})
    endif()

    # Add include directories for the compiler to find project headers
    target_include_directories(${target_name} PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}  # For local headers
        ${CMAKE_SOURCE_DIR}/src      # For common headers like ../common/fits_api.h
    )
    
    # Make include directories available to targets that link to this library
    target_include_directories(${target_name} INTERFACE 
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()


add_subdirectory(src)
add_subdirectory(tests)