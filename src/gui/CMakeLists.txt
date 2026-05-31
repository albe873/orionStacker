find_package(PkgConfig REQUIRED)
pkg_check_modules(GTKMM REQUIRED IMPORTED_TARGET gtkmm-4.0)

add_executable(myapp main.cpp helloworld.cpp)

# Link using modern CMake imported target (preferred)
target_link_libraries(myapp PRIVATE PkgConfig::GTKMM)