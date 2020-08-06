cmake_minimum_required(VERSION 3.10)
cmake_policy(VERSION 3.10)

project(se-apps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Fucntion to create the executables given the name of one of the compiled
# se-denseslam libraries.
function(add_version lib_name libraries)

  if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__APPLE__")
    set(main_common_libraries
      ${libraries})
  else(APPLE)
    set(main_common_libraries
      ${libraries}
      rt)
  endif(APPLE)

  set(main_common_compile_flags "")

  set(reader_sources
    src/reader_base.cpp
	src/reader_iclnuim.cpp
	src/reader_openni.cpp
	src/reader_raw.cpp
	src/reader.cpp
  )

  if (OPENNI2_FOUND)
    include_directories(${OPENNI2_INCLUDE_PATHS})
    set(main_common_libraries
      ${main_common_libraries}
      ${OPENNI2_LIBRARIES})
    set(main_common_compile_flags "-DSE_USE_OPENNI ${main_common_compile_flags}")
  endif(OPENNI2_FOUND)

  #  ------ MAIN --------
  add_executable(${lib_name}-main
    src/main.cpp
    src/PowerMonitor.cpp
    ${reader_sources}
  )
  target_include_directories(${lib_name}-main PUBLIC include)
  target_link_libraries(${lib_name}-main
    PUBLIC
      ${lib_name}
      ${main_common_libraries}
  )
  set_target_properties(${lib_name}-main PROPERTIES COMPILE_FLAGS
    "${main_common_compile_flags}")
  # Compile with GUI support
  if (GLUT_FOUND)
    target_compile_definitions(${lib_name}-main PUBLIC -DSE_GLUT)
    target_link_libraries(${lib_name}-main
      PUBLIC
        ${GLUT_LIBRARIES}
        ${OPENGL_LIBRARIES}
    )
    if(NOT APPLE)
      target_link_libraries(${lib_name}-main PUBLIC GL)
    endif(NOT APPLE)
  endif(GLUT_FOUND)
  # Use floats in srl_projection
  target_compile_definitions(${lib_name}-main
    PUBLIC
  	SRL_FLOAT_T=float
  )

  #  ------ QT --------
  if (Qt5OpenGL_FOUND)

    file(GLOB qt_srcs qt/*)

    QT5_ADD_RESOURCES( ResourceSources
      "qt/images.qrc")
    set(qt_srcs
      ${qt_srcs}
      ${ResourceSources})
    set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
    set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR ON)

    include_directories(qt)
    add_definitions(-D_FILE_OFFSET_BITS=64 -fPIC)

    add_executable(${lib_name}-qt
      src/main.cpp
      src/PowerMonitor.cpp
      ${qt_srcs}
      ${reader_sources})
    target_include_directories(${lib_name}-qt PUBLIC
      include)

    add_definitions(${Qt5Widgets_DEFINITIONS})
    add_definitions(${Qt5OpenGL_DEFINITIONS})
    add_definitions(${Qt5PrintSuppoer_DEFINITIONS})
    include_directories(${Qt5Widgets_INCLUDE_DIRS})
    include_directories(${Qt5OpenGL_INCLUDE_DIRS})
    include_directories(${Qt5PrintSupport_INCLUDE_DIRS})
    include_directories(${Qt5Gui_INCLUDE_DIRS})

    set_target_properties(${lib_name}-qt PROPERTIES COMPILE_FLAGS
      "${main_common_compile_flags} -D__QT__")

    if(APPLE)
      target_link_libraries(${lib_name}-qt
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::PrintSupport
        ${main_common_libraries}
        ${lib_name}
        ${GLUT_LIBRARIES}
        ${OPENGL_LIBRARIES})
    else(APPLE)
      target_link_libraries(${lib_name}-qt
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::PrintSupport
        ${main_common_libraries}
        ${lib_name}
        ${GLUT_LIBRARIES}
        ${OPENGL_LIBRARIES}
        GL)
    endif(APPLE)
    # Use floats in srl_projection
    target_compile_definitions(${lib_name}-qt
      PUBLIC
    	SRL_FLOAT_T=float
    )
  endif(Qt5OpenGL_FOUND)

endfunction()



# Find optional packages.
find_package(GLUT)
find_package(OpenGL)
find_package(OpenNI2)
find_package(PkgConfig)
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5Gui)
find_package(Qt5OpenGL)
find_package(Qt5PrintSupport)

# Set the common libraries to link.
set(common_libraries stdc++ stdc++fs ${SUPEREIGHT_CORE_LIBS})

# Create the executables for each of the compiled se-denseslam libraries.
foreach(lib_name ${SUPEREIGHT_DENSESLAM_LIBS})
  add_version(${lib_name} "${common_libraries}")
endforeach()

