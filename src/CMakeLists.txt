
# Lib for math.
set(LIB_NAME math)
file(GLOB SOURCE_FILES "${LIB_NAME}/*.cc")
add_library(${LIB_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${LIB_NAME}  ${PUPIL_LIBRARIES})
set(EXTRA_LIBS ${EXTRA_LIBS} ${LIB_NAME})

# Lib for core.
set(LIB_NAME core)
file(GLOB SOURCE_FILES "${LIB_NAME}/*.cc")
add_library(${LIB_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${LIB_NAME}  ${PUPIL_LIBRARIES})
set(EXTRA_LIBS ${EXTRA_LIBS} ${LIB_NAME})

# Lib for inputs.
set(LIB_NAME inputs)
file(GLOB SOURCE_FILES "${LIB_NAME}/*.cc")
add_library(${LIB_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${LIB_NAME}  ${PUPIL_LIBRARIES})
set(EXTRA_LIBS ${EXTRA_LIBS} ${LIB_NAME})

# Apps.
add_executable(demo app/demo.cc)
target_link_libraries(demo ${PUPIL_LIBRARIES} ${EXTRA_LIBS})