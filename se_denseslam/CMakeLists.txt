cmake_minimum_required(VERSION 3.5...3.16)

find_package(octomap)
find_package(ompl REQUIRED)

# Generate the appropriate include line for each voxel implementation.
foreach(VOXEL_IMPL ${SE_VOXEL_IMPLS})
  string(APPEND VOXEL_IMPL_INCLUDES
    "#include <se/voxel_implementations/${VOXEL_IMPL}/${VOXEL_IMPL}.hpp>\n")
endforeach()

# Generate header file including each voxel implementation.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/se/voxel_implementations.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/se/voxel_implementations.hpp"
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/include/se/voxel_implementations.hpp"
  DESTINATION include/se
)



# Create library ##############################################################
# Iterate over each voxel and sensor implementaion and compile into an
# se-denseslam library.
foreach(SENSOR_IMPL ${SE_SENSOR_IMPLS})
  # Prepare the library name
  string(TOLOWER ${SENSOR_IMPL} SENSOR_IMPL_LC)
  set(LIB_NAME "se-denseslam-${SENSOR_IMPL_LC}")

  # Add the static library
  add_library(${LIB_NAME} STATIC
    "./src/preprocessing.cpp"
    "./src/tracking.cpp"
    "./src/rendering.cpp"
    "./src/depth_utils.cpp"
    "./src/object.cpp"
    "./src/object_rendering.cpp"
    "./src/object_utils.cpp"
    "./src/ptp/OccupancyWorld.cpp"
    "./src/ptp/ProbCollisionChecker.cpp"
    "./src/ptp/SafeFlightCorridorGenerator.cpp"
    "./src/morton_sampling_tree.cpp"
    "./src/entropy.cpp"
    "./src/lod.cpp"
    "./src/dist.cpp"
    "./src/completion.cpp"
    "./src/path.cpp"
    "./src/candidate_view.cpp"
    "./src/pose_history.cpp"
    "./src/pose_grid_history.cpp"
    "./src/pose_mask_history.cpp"
    "./src/pose_vector_history.cpp"
    "./src/exploration_utils.cpp"
    "./src/single_path_exploration_planner.cpp"
    "./src/exploration_planner.cpp"
    "./src/DenseSLAMSystem.cpp"
  )
  target_include_directories(${LIB_NAME} BEFORE
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
      $<INSTALL_INTERFACE:include>
      ${OMPL_INCLUDE_DIRS}
  )
  target_link_libraries(${LIB_NAME}
    PUBLIC
      ${OMPL_LIBRARIES}
	  FI::FrustumIntersector
  )
  set_target_properties(${LIB_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
  if(DEFINED OCTOMAP_INCLUDE_DIRS)
    target_include_directories(${LIB_NAME} PUBLIC ${OCTOMAP_INCLUDE_DIRS})
    target_link_libraries(${LIB_NAME} PUBLIC ${OCTOMAP_LIBRARIES})
    target_compile_definitions(${LIB_NAME} PUBLIC SE_OCTOMAP=1)
  else()
    target_compile_definitions(${LIB_NAME} PUBLIC SE_OCTOMAP=0)
  endif()
  # Define the voxel implementation macro as the first one in the list
  list(GET SE_VOXEL_IMPLS 0 VOXEL_IMPL_MACRO)
  target_compile_definitions(${LIB_NAME}
    PUBLIC
      SE_VOXEL_IMPLEMENTATION=${VOXEL_IMPL_MACRO}
      SE_OBJECT_VOXEL_IMPLEMENTATION=MultiresTSDF
  )

  foreach(VOXEL_IMPL ${SE_VOXEL_IMPLS})
    # Link with the voxel implementation
    target_link_libraries(${LIB_NAME}
      PUBLIC
        SE::VoxelImpl${VOXEL_IMPL}${SENSOR_IMPL}
    )
  endforeach()

  # Add an alias so that the library can be used inside the build tree, e.g.
  # when testing
  add_library(SE::DenseSLAM${SENSOR_IMPL} ALIAS ${LIB_NAME})
  # This is required so that the exported target has the name
  # DenseSLAMVoxelSensor and not se-denseslam-voxel-sensor
  set_target_properties(${LIB_NAME} PROPERTIES EXPORT_NAME DenseSLAM${SENSOR_IMPL})



  # Install library #########################################################
  set(SUBPACKAGE_NAME SEDenseSLAM${SENSOR_IMPL})
  # Export the targets to a script
  install(EXPORT ${LIB_NAME}-targets
    FILE
      ${SUBPACKAGE_NAME}Targets.cmake
    NAMESPACE
      SE::
    DESTINATION
      "${CMAKE_INSTALL_CMAKEDIR}"
  )
  # Install the headers
  install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  # Install the static library
  install(TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )

  # Export from the build tree ##############################################
  export(EXPORT ${LIB_NAME}-targets
      FILE "${CMAKE_CURRENT_BINARY_DIR}/${SUBPACKAGE_NAME}Targets.cmake"
      NAMESPACE SE::
  )
endforeach()



# Test library ################################################################
if(SE_BUILD_TESTS)
  add_subdirectory(test)
endif()

