cmake_minimum_required(VERSION 3.5...3.16)

find_package(OpenNI2)
find_package(GLUT)
find_package(OpenGL)

set(READER_SRC
  src/reader_base.cpp
  src/reader_interiornet.cpp
  src/reader_newercollege.cpp
  src/reader_openni.cpp
  src/reader_raw.cpp
  src/reader_tum.cpp
  src/reader.cpp
)

# Create executables ##########################################################
# Iterate over each voxel and sensor implementaion and compile the executables
# for it.
foreach(SENSOR_IMPL ${SE_SENSOR_IMPLS})
  # Convert the implementation names to lowercase
  string(TOLOWER ${SENSOR_IMPL} SENSOR_IMPL_LC)

  # Main executable #########################################################
  set(EXE_NAME "se-denseslam-${SENSOR_IMPL_LC}-main")
  add_executable(${EXE_NAME}
    "src/main.cpp"
    "src/montage.cpp"
    ${READER_SRC}
  )
  target_include_directories(${EXE_NAME} BEFORE PRIVATE include)
  target_link_libraries(${EXE_NAME}
    PRIVATE
      SE::DenseSLAM${SENSOR_IMPL}
  )
  # Allow handling large files in 32-bit systems
  target_compile_definitions(${EXE_NAME} PRIVATE "-D_FILE_OFFSET_BITS=64")
  set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
  # Compile with GUI support
  if(SE_BUILD_GLUT_GUI AND GLUT_FOUND)
    target_link_libraries(${EXE_NAME}
      PRIVATE
        ${GLUT_LIBRARIES}
        ${OPENGL_LIBRARIES}
    )
    target_compile_definitions(${EXE_NAME} PRIVATE "-DSE_GLUT")
    if(NOT APPLE)
      target_link_libraries(${EXE_NAME} PRIVATE GL)
    endif()
  endif()
  # Compile with OpenNI2 support
  if(OPENNI2_FOUND)
    target_include_directories(${EXE_NAME} PRIVATE ${OPENNI2_INCLUDE_PATHS})
    target_link_libraries(${EXE_NAME} PRIVATE ${OPENNI2_LIBRARIES})
    target_compile_definitions(${EXE_NAME} PRIVATE "-DSE_USE_OPENNI")
  endif()
endforeach()



# Executable showing voxel data sizes
set(EXE_NAME "se-voxel-size")
add_executable(${EXE_NAME} "src/voxel_size.cpp")
target_include_directories(${EXE_NAME} BEFORE PRIVATE ../se_voxel_impl/include)
target_link_libraries(${EXE_NAME} PRIVATE SE::Shared)
target_compile_definitions(${EXE_NAME} PUBLIC SE_SENSOR_IMPLEMENTATION=PinholeCamera)



# Morton conversion tool
set(EXE_NAME "se-morton-tool")
add_executable(${EXE_NAME} "src/morton_tool.cpp")
target_link_libraries(${EXE_NAME} PRIVATE SE::Core)



# Test ########################################################################
if(SE_BUILD_TESTS)
  add_subdirectory(test)
endif()

