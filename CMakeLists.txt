cmake_minimum_required(VERSION 2.8.3)

SET(CMAKE_BUILD_TYPE "Release")

project(pgo_livox)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(ROS_DEP roscpp rospy std_msgs sensor_msgs nav_msgs cv_bridge image_transport)
find_package(OpenCV 4.2 REQUIRED)
find_package(PCL 1.12 REQUIRED)
find_package(OpenMP REQUIRED)

find_package(catkin REQUIRED COMPONENTS
       ${ROS_DEP}
        )
find_package(GTSAM REQUIRED QUIET)


catkin_package(CATKIN_DEPENDS ${ROS_DEP})

include_directories(
        /usr/local/include/eigen3
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIR}
        include
)

add_executable(pgo_livox
        include/util.h
        include/tictoc.h include/tictoc.cpp
        include/PointCloudMatcher.h  include/PointCloudMatcher.cpp
        include/Optimize.h include/Optimize.cpp
        src/pgo_livox.cpp)
target_compile_options(pgo_livox
        PRIVATE ${OpenMP_CXX_FLAGS}
        )

target_link_libraries(
        pgo_livox
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
         ${OpenMP_CXX_FLAGS}
        gtsam
)

