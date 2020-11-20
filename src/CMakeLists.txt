
add_library(lib_geometry geometry.cc)
target_link_libraries(lib_geometry  ${PUPIL_LIBRARIES})
set(PUPIL_LIBRARIES ${PUPIL_LIBRARIES} lib_geometry)

add_library(lib_union_find union_find.cc)
target_link_libraries(lib_union_find  ${PUPIL_LIBRARIES})
set(PUPIL_LIBRARIES ${PUPIL_LIBRARIES} lib_union_find)

add_library(lib_pupil_detector pupil_detector.cc)
target_link_libraries(lib_pupil_detector  ${PUPIL_LIBRARIES})
set(PUPIL_LIBRARIES ${PUPIL_LIBRARIES} lib_pupil_detector)


add_library(lib_solver solver.cc)
target_link_libraries(lib_solver  ${PUPIL_LIBRARIES})
set(PUPIL_LIBRARIES ${PUPIL_LIBRARIES} lib_solver)

add_library(lib_camera camera.cc)
target_link_libraries(lib_camera  ${PUPIL_LIBRARIES})
set(PUPIL_LIBRARIES ${PUPIL_LIBRARIES} lib_camera)

add_library(lib_tracker eyeball_tracker.cc)
target_link_libraries(lib_tracker  ${PUPIL_LIBRARIES})
set(PUPIL_LIBRARIES ${PUPIL_LIBRARIES} lib_tracker)

add_executable(demo demo.cc)
target_link_libraries(demo ${PUPIL_LIBRARIES})