cmake_minimum_required(VERSION 3.8)
project(tidy_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Reset.srv"
  "srv/Step.srv"
  "srv/GoHome.srv"
  "srv/Render.srv"
  "srv/Retarget.srv"
  "srv/Plan.srv"
  "srv/Point.srv"
  "srv/Execute.srv"
  "msg/Frame.msg"
  DEPENDENCIES std_msgs sensor_msgs
)
ament_package()
