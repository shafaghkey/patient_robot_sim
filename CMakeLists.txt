cmake_minimum_required(VERSION 3.0.2)
project(patient_robot_sim)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs
)

## Mark executable scripts (Python) for installation
catkin_install_python(PROGRAMS
  controllers/human_controller.py
  controllers/human_controller_paper.py
  controllers/human_controller_role.py
  controllers/human_controller_role_legacy.py
  controllers/gripper_controller.py
  controllers/robot_controller.py
  controllers/robot_controller_0.py
  controllers/robot_controller_SCA.py
  controllers/robot_controller-CBF.py
  controllers/robot_controller_role.py
  controllers/robot_controller_role_legacy.py
  controllers/robot_nogripper_controller.py
  scenarios/hri_sim.py
  scenarios/multi_robot_control.py
  scenarios/patient_robot.py
  scenarios/patient_robot_2D_ROM_role.py
  scenarios/patient_robot_ROM.py
  scenarios/patient_robot_ROM_role.py
  scenarios/patient_robot_gripperSeparateURDF.py
  scenarios/patient_robot_nogripper.py
  tasks/controller_paper_task_impaired.py
  tasks/controller_task_impaired_legacy.py
  tasks/controller_task_impaired_passive.py
  tasks/controller_task_impaired_state_machine.py
  nodes/human_sim_node.py
  nodes/hri_sim_ros_patient_impaired.py
  nodes/human_publisher.py
  scripts/set_initial_joints.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (launch, config, assets, python modules)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY agents/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/agents
)

# Install shared Python packages
install(DIRECTORY utils/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/utils
)

install(DIRECTORY controllers/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/controllers
)

install(DIRECTORY scenarios/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/scenarios
)

install(DIRECTORY tasks/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/tasks
)
