add_subdirectory(common)
add_subdirectory(calibration)
add_subdirectory(debayer)

# GUI subdirectory — requires gtkmm-4.0, skipped automatically if not found.
# Set -DBUILD_GUI=OFF to disable explicitly.
option(BUILD_GUI "Build the GUI application (requires gtkmm-4.0)" ON)
if (BUILD_GUI)
    add_subdirectory(gui)
endif()

add_subdirectory(stacker)
add_subdirectory(star_finder)