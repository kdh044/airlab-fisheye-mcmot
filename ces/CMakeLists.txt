cmake_minimum_required(VERSION 3.0.2)
project(ces)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  visualization_msgs
  message_filters
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# 파이썬 스크립트 설치 (MOT/scripts 폴더 안의 파일 인식)
catkin_install_python(PROGRAMS
  MOT/scripts/fisheye_detection.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
