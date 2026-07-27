# Optional GUI — only build if gtkmm-4.0 and its dependencies are available.
find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
    pkg_check_modules(GTKMM IMPORTED_TARGET gtkmm-4.0)
endif()

if (GTKMM_FOUND)
    add_executable(myapp main.cpp helloworld.cpp)
    target_link_libraries(myapp PRIVATE PkgConfig::GTKMM)
else()
    message(STATUS "gtkmm-4.0 not found — skipping GUI build")
endif()