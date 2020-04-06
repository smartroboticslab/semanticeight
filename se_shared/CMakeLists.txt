cmake_minimum_required(VERSION 3.10)

find_package(Eigen3 REQUIRED)
find_package(LodePNG REQUIRED)

# Library containing code common for all supereight modules.
add_library(se-shared STATIC
  src/str_utils.cpp
  src/image_utils.cpp
)
target_include_directories(se-shared
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(se-shared
  PUBLIC
    LodePNG::lodepng
)
install(TARGETS se-shared
  EXPORT supereightTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
# Populate the list of libraries that are part of se-shared.
list(APPEND SUPEREIGHT_SHARED_LIBS se-shared)



# Install the headers for all libraries in se-shared.
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION include
)

# Make the list of se-shared libraries available to the parent file.
set(SUPEREIGHT_SHARED_LIBS ${SUPEREIGHT_SHARED_LIBS} PARENT_SCOPE)

