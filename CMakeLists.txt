cmake_minimum_required(VERSION 3.28)

project(cuda_stacker)

# Default to Release to avoid unoptimized GPU builds when CMAKE_BUILD_TYPE is omitted.
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui features2d calib3d videoio flann)

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit found: ${CUDAToolkit_VERSION}")
    if (CPU_ONLY)
        message(STATUS "Compiling only CPU version")
    elseif (ROCM)
        # Help CMake discover HIP on standard ROCm installations.
        if (NOT DEFINED HIP_DIR OR NOT HIP_DIR OR HIP_DIR MATCHES "-NOTFOUND$")
            set(_ROCM_PREFIX_HINTS "/opt/rocm")
            if (DEFINED ENV{ROCM_PATH})
                list(APPEND _ROCM_PREFIX_HINTS "$ENV{ROCM_PATH}")
            endif()

            file(GLOB _ROCM_GLOB_HINTS LIST_DIRECTORIES true "/opt/rocm-*")
            list(APPEND _ROCM_PREFIX_HINTS ${_ROCM_GLOB_HINTS})

            foreach(_rocm_prefix IN LISTS _ROCM_PREFIX_HINTS)
                if (EXISTS "${_rocm_prefix}/lib/cmake/hip/hip-config.cmake")
                    set(HIP_DIR "${_rocm_prefix}/lib/cmake/hip")
                    list(PREPEND CMAKE_PREFIX_PATH "${_rocm_prefix}")
                    break()
                endif()
            endforeach()
        endif()

        find_package(HIP REQUIRED CONFIG)
        enable_language(HIP)
        set(HIP_ARCHITECTURES "gfx1030")
        message(STATUS "Compiling with ROCM support")
    else()
        enable_language(CUDA)
        set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86 87 89 90)
        message(STATUS "Compiling with CUDA support")
    endif()
else()
    message(STATUS "CUDAToolkit not found. Comiling only CPU version")
endif()


function(add_cuda_hip_executable target_name cuda_source)
    set(extra_includes "")
    set(extra_include_dirs "")
    foreach(arg IN LISTS ARGN)
        list(APPEND extra_include_dirs ${arg})
        list(APPEND extra_includes "-I${arg}")
    endforeach()

    # Inherit include directories from the caller directory.
    get_directory_property(_caller_include_dirs INCLUDE_DIRECTORIES)
    foreach(inc_dir IN LISTS _caller_include_dirs)
        list(APPEND extra_include_dirs ${inc_dir})
        list(APPEND extra_includes "-I${inc_dir}")
    endforeach()

    # Auto-propagate OpenCV includes when available in the caller scope.
    if (DEFINED OpenCV_INCLUDE_DIRS)
        foreach(cv_inc IN LISTS OpenCV_INCLUDE_DIRS)
            list(APPEND extra_include_dirs ${cv_inc})
            list(APPEND extra_includes "-I${cv_inc}")
        endforeach()
    endif()

    if(ROCM)
        # Hipify the CUDA file
        set(HIPIFIED_FILE ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.hip)
        add_custom_command(
            OUTPUT ${HIPIFIED_FILE}
            COMMAND hipify-clang ${CMAKE_CURRENT_SOURCE_DIR}/${cuda_source} -o ${HIPIFIED_FILE}
                    ${_HIPIFY_CUDA_PATH_ARG}
                    -I${CMAKE_SOURCE_DIR}/src/common/include
                    -I${CMAKE_SOURCE_DIR}/src/debayer/include
                    -I${CMAKE_SOURCE_DIR}/src/calibration/include
                    -I${CMAKE_SOURCE_DIR}/src/stacker 
                    -I${CMAKE_SOURCE_DIR}/src/star_finder
                    ${extra_includes}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${cuda_source}
            COMMENT "Hipifying ${cuda_source}"
        )
        add_executable(${target_name} ${HIPIFIED_FILE})

        # Add include directories for the compiler to find project headers
        # CMAKE_CURRENT_SOURCE_DIR refers to the directory of the CMakeLists.txt that *calls* this function
        target_include_directories(${target_name} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}  # For local headers like device_alfa_sigma.h
            ${CMAKE_SOURCE_DIR}/src      # For common headers like ../common/fits_api.h
            ${extra_include_dirs}
        )
    else()
        add_executable(${target_name} ${cuda_source})
        target_include_directories(${target_name} PRIVATE
            ${extra_include_dirs}
        )
    endif()
endfunction()

function(add_cuda_hip_library target_name cuda_source)
    set(extra_includes "")
    get_directory_property(_caller_include_dirs INCLUDE_DIRECTORIES)
    foreach(inc_dir IN LISTS _caller_include_dirs)
        list(APPEND extra_includes "-I${inc_dir}")
    endforeach()

    if(ROCM)
        # Hipify the CUDA file
        set(HIPIFIED_FILE ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.hip)
        add_custom_command(
            OUTPUT ${HIPIFIED_FILE}
            COMMAND hipify-clang ${CMAKE_CURRENT_SOURCE_DIR}/${cuda_source} -o ${HIPIFIED_FILE}
                    ${_HIPIFY_CUDA_PATH_ARG}
                    -I${CMAKE_SOURCE_DIR}/src 
                    -I${CMAKE_SOURCE_DIR}/src/common/include
                    ${extra_includes}
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
add_subdirectory(test)