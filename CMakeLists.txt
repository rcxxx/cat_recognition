cmake_minimum_required(VERSION 3.23)
project(cat_recognition)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add include dir
include_directories(
        ${PROJECT_SOURCE_DIR}/modules
        ${PROJECT_SOURCE_DIR}/include
)

# Dependence lib
## Find and add opencv as required
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

## Find and add libtorch as required
set(CMAKE_PREFIX_PATH "~/WorkSpace/code/libtorch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
link_libraries(${TORCH_LIBRARIES})

# Add Subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/modules/yolov5)
add_subdirectory(${PROJECT_SOURCE_DIR}/modules/resnet)


file(GLOB_RECURSE mains RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
foreach(mainfile IN LISTS mains)
    # Get file name without directory
    get_filename_component(mainname ${mainfile} NAME_WE)
    add_executable(${mainname} ${mainfile})
    # Target
    target_link_libraries(${mainname}
            ${OpenCV_LIBS}
            ${TORCH_LIBRARIES}
            yolov5
            resnet)
endforeach()

#add_executable(cat_recognition src/main.cpp)
