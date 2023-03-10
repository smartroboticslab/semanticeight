cmake_minimum_required(VERSION 3.5...3.16)

project(frustum-intersector VERSION 0.1.0 LANGUAGES CXX)

# Create library ##############################################################
find_package(Eigen3 REQUIRED)
find_package(CGAL REQUIRED)

add_library(${PROJECT_NAME} STATIC "src/frustum_intersector.cpp")
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    Eigen3::Eigen
    CGAL::CGAL
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS NO
)

# Add an alias so that the library can be used inside the build tree, e.g. when
# testing
add_library(FI::FrustumIntersector ALIAS ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME FrustumIntersector)



# Install the library #########################################################
set(PACKAGE_NAME FrustumIntersector)
# Make sure that custom modules are found
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)
# Install includes
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
# The directory where the .cmake files will be installed
if(WIN32 AND NOT CYGWIN)
  set(CMAKE_INSTALL_CMAKEDIR "${PACKAGE_NAME}/cmake")
elseif(APPLE)
  set(CMAKE_INSTALL_CMAKEDIR "${PACKAGE_NAME}.framework/Resources/CMake")
else()
  set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}")
endif()

# Create a Config.cmake file
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PACKAGE_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
)

# Create a ConfigVersion.cmake file
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Export the targets to a script
install(EXPORT ${PROJECT_NAME}-targets
  FILE
    ${PACKAGE_NAME}Targets.cmake
  NAMESPACE
    FI::
  DESTINATION
    "${CMAKE_INSTALL_CMAKEDIR}"
)
# Install the headers
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
# Install the static library
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
# Install the Config and ConfigVersion files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}ConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
)



# Export from the build tree ##################################################
export(EXPORT ${PROJECT_NAME}-targets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Targets.cmake"
    NAMESPACE FI::
)
# Register package in user's package registry
export(PACKAGE ${PACKAGE_NAME})

