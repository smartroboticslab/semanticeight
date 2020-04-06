cmake_minimum_required(VERSION 3.10)

# lodepng third-party shared library.
add_library(lodepng STATIC
  src/lodepng.cpp
)
target_include_directories(lodepng PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
install(TARGETS lodepng
  EXPORT supereightTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
# Populate the list of libraries that are part of se-shared.
list(APPEND SUPEREIGHT_SHARED_LIBS lodepng)



# Library containing code common for all supereight modules.
find_package(Eigen3 REQUIRED)
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
    lodepng
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
  DESTINATION include)

# Make the list of se-shared libraries available to the parent file.
set(SUPEREIGHT_SHARED_LIBS ${SUPEREIGHT_SHARED_LIBS} PARENT_SCOPE)

