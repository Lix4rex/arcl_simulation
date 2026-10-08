cmake_minimum_required(VERSION 3.8)
project(arcl_simulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp)
find_package(std_msgs)
find_package(robot_msgs)
find_package(sensor_msgs)
find_package(visualization_msgs)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)


add_executable(robot_control src/robot_control.cpp)
target_include_directories(robot_control PRIVATE ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(robot_control rclcpp std_msgs robot_msgs geometry_msgs sensor_msgs nav_msgs)

add_executable(ekf_solo src/ekf_solo.cpp)
target_include_directories(ekf_solo PRIVATE ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(ekf_solo rclcpp sensor_msgs geometry_msgs visualization_msgs robot_msgs) 

add_executable(table_markers src/table_markers.cpp)
ament_target_dependencies(table_markers rclcpp visualization_msgs)

install(TARGETS robot_control ekf_solo table_markers
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config launch rviz urdf worlds 
  DESTINATION share/${PROJECT_NAME})



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()