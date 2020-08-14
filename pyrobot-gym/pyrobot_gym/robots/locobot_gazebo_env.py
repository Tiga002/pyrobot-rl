import numpy as np
import rospy
from gazebo_msgs.srv import GetWorldProperties, GetModelState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from pyrobot_gym.core.openai_ros_common import ROSLauncher
from pyrobot_gym.core import robot_gazebo_env
import sys
import time
#import moveit_commander
#import moveit_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
from pyrobot import Robot
import tf

"""
Boundaries of the Configuration Space
"""
BOUNDS_CEILLING = .45
BOUNDS_FLOOR = .08
BOUNDS_LEFTWALL = .45
BOUNDS_RIGHTWALL = -.45
BOUNDS_FRONTWALL = .5
BOUNDS_BACKWALL = -.13
JOINT_1_LIMIT = 1.57
JOINT_2_LIMIT = 1.57
JOINT_3_LIMIT = 1.57
JOINT_4_LIMIT = 1.57
JOINT_5_LIMIT = 1.57

class LocoBotGazeboEnv(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self, ros_ws_abspath, ros_ws_src_path):
        rospy.logdebug("======= In LocobotGazeboEnv")

        # launch the locobot and world spawn launch file
        #ROSLauncher(rospackage_name="locobot_gazebo",
        #            launch_file_name="put_robot_in_world_HER.launch",
        #            ros_ws_abspath=ros_ws_abspath,
        #            ros_ws_src_path=ros_ws_src_path)

        # Object controls all objects' positions !!!
        #self.obj_positions = Object_Position()

        self.controllers_list = []

        self.robot_name_space = ""
        self.reset_controls = False

        super(LocoBotGazeboEnv, self).__init__(controllers_list=self.controllers_list,
                                               robot_name_space=self.robot_name_space,
                                               reset_controls=False,
                                               start_init_physics_parameters=True,
                                               reset_world_or_sim="WORLD")

        rospy.logdebug("Goint to create a PyRobot Instance")
        self.robot = Robot('locobot')
        print("ABCBSDASKDJASDLKASJDALKSDJALSKDJALSKDJASKDJASLDKAJSDLKASJDLAKSDJALSKDJ")
        rospy.logdebug("PyRobot Instance created")

        # We start all the ROS related Subscriber and Publishers

        self.JOINT_STATES_SUBSCRIBE_TOPIC = '/joint_states'
        self.joint_names = ["joint_1",
                            "joint_2",
                            "joint_3",
                            "joint_1",
                            "joint_4",
                            "joint_5",
                            "joint_6",
                            "joint_7"]
        self.gazebo.unpauseSim()
        self._check_all_systems_ready()

        self.joint_states_sub = rospy.Subscriber(
            self.JOINT_STATES_SUBSCRIBE_TOPIC, JointState, self.joints_callback)
        self.joints = JointState()

        # Start MoveIt Service
        #self.move_locobot_object = MoveLocoBot()



        # Wait until LocoBot reached its StartUp positions
        #self.wait_locobot_ready()

        self.gazebo.pauseSim()
        rospy.logdebug("======= Out LocobotGazeboEnv")

    # RobotGazeboEnv Virtual Methods overrided here ...
    # --------------------------------------------------

    def _check_all_systems_ready(self):
        """
        Check all the sensors, publishers, and other simulation system elemets
        are ready for operation
        """
        self._check_all_sensors_ready()
        return True

    # LocobotGazeboEnv Methods
    # -----------------------------------------------

    def _check_all_sensors_ready(self):
        self._check_joint_states_ready()
        rospy.logdebug("All Sensors are ready ---")

    def _check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                # Subscribe the Joint States while Sim is unpaused
                self.joints = rospy.wait_for_message(
                self.JOINT_STATES_SUBSCRIBE_TOPIC, JointState, timeout=1.0)
                #rospy.logdebug(
                #    "Current " + str(self.JOINT_STATES_SUBSCRIBE_TOPIC)
                #    + "Ready => " + str(self.joints))
            except:
                rospy.logerr(
                "Current "+str(self.JOINT_STATES_SUBSCRIBE_TOPIC)+" not ready yet, retrying....")
        return self.joints

    def joints_callback(self, data):
        """
        Callback function for joint_states_sub
        """
        self.joints = data

    def get_joints_names(self):
        return self.joint.names

    def _set_startup_position(self):
        result = self.robot.arm.go_home()
        print('Reseting to startup position ~~~~~~~~')
        return result

    def set_trajectory_ee(self, action):
        """
        Sets the Pose of the End Effector based on the action.
        The action contains the desired position and orientation of the End Effector
        See create_action
        """
        """
        # Set up a trajectory message to publish
        ee_target = geometry_msgs.msg.Pose()
        ee_target.orientation.x = -0.707
        ee_target.orientation.y = 0.0
        ee_target.orientation.z = 0.707
        ee_target.orientation.w = 0.001

        ee_target.position.x = action[0]
        ee_target.position.y = action[1]
        ee_target.position.z = action[2]
        """
        target_pose = {'position':np.array([action[0], action[1], action[2]]),
                       'orientation':np.array([0.245, 0.613, -0.202, 0.723])}

        # This will drive the end effector to the target pos+ori
        #result = self.move_locobot_object.ee_traj(ee_target)
        result = self.robot.arm.set_ee_pose(**target_pose)
        #time.sleep(1)
        return result

    def set_trajectory_joints(self, joint_positions):
        """
        Setting the Initial Joint States
        """
        positions_array = [None] * 5
        positions_array[0] = joint_positions[0]
        positions_array[1] = joint_positions[1]
        positions_array[2] = joint_positions[2]
        positions_array[3] = joint_positions[3]
        positions_array[4] = joint_positions[4]
        # Check #1
        # Check is it within the joint limits
        conditions = [positions_array[0] <= JOINT_1_LIMIT, positions_array[0] >= -JOINT_1_LIMIT,
                      positions_array[1] <= JOINT_2_LIMIT, positions_array[1] >= -JOINT_2_LIMIT,
                      positions_array[2] <= JOINT_3_LIMIT, positions_array[2] >= -JOINT_3_LIMIT,
                      positions_array[3] <= JOINT_4_LIMIT, positions_array[3] >= -JOINT_4_LIMIT,
                      positions_array[4] <= JOINT_5_LIMIT, positions_array[4] >= -JOINT_5_LIMIT]
        rospy.logdebug('joints_conditions = {}'.format(conditions))
        violated_joint_limit = False
        for condition in conditions:
            if not condition:
                violated_joint_limit = True
                break

        if violated_joint_limit == True:
            rospy.logerr('Desired Joint Positions are out of Joint Limits')
            result = False
        else:
            result = self.robot.arm.set_joint_positions(positions_array, plan=False)
            """
            # Check #2
            # Perform a Forward Kinematics on the new desired joint positions
            # To check if it is within the Configuration Space
            fk_pos = self.robot.arm.compute_fk_position(np.array(positions_array), 'gripper_link')[0]
            rospy.logdebug('fk_pos = {}'.format(fk_pos))
            conditions = [fk_pos[0] <= BOUNDS_FRONTWALL, fk_pos[0] >= BOUNDS_BACKWALL,
                          fk_pos[1] <= BOUNDS_LEFTWALL, fk_pos[1] >= BOUNDS_RIGHTWALL,
                          fk_pos[2] <= BOUNDS_CEILLING, fk_pos[2] >= BOUNDS_FLOOR]
            rospy.logdebug('fk_eff_conditions = {}'.format(conditions))
            violated_boundary = False
            for condition in conditions:
                if not condition:
                    violated_boundary = True
                    break
            if violated_boundary == True:
                rospy.logerr('Desired gripper place is out of the configuration space Boundaries')
                result = False
            else:
                rospy.logdebug('Desired gripper place within the configuration space Boundaries')
                result = self.robot.gripper.close()
                result = self.robot.arm.set_joint_positions(positions_array, plan=False)
            """
        #time.sleep(1)
        # Only in play
        #if result == False:
        #    self.robot.arm.go_home()
        #    rospy.logdebug('Reset back to the startup position and start again ~~~')
        """For showing difference
        result = self.robot.gripper.close()
        result = self.robot.arm.set_joint_positions(positions_array, plan=False)
        """
        return result

    def set_trajectory_initial_joints(self, initial_qpos):
        """
        Setting the Initial Joint States
        """
        positions_array = [None] * 5
        positions_array[0] = initial_qpos["joint_1"]
        positions_array[1] = initial_qpos["joint_2"]
        positions_array[2] = initial_qpos["joint_3"]
        positions_array[3] = initial_qpos["joint_4"]
        positions_array[4] = initial_qpos["joint_5"]

        #self.move_locobot_object.joint_traj(positions_array)
        result = self.robot.gripper.close()
        result = self.robot.arm.set_joint_positions(positions_array, plan=False)
        #time.sleep(1)

        return True

    def create_action(self, position, orientation):
        """
        position = [x,y,z]
        orientation = [x,y,z,w] #Quaternion
        """
        gripper_target_position = np.array(position)
        gripper_target_rotation = np.array(orientation)
        action = np.concatenate([gripper_target_position, gripper_target_rotation])

        return action

    def create_joints_dict(self, joints_positions):
        """
        Based on the Order of the positions, they will be assigned to its joint name
        names_in_order:
        joint_1: 0.0
        joint_2: 0.0
        joint_3: -1.5
        joint_4: 0.0
        joint_5: 1.5
        joint_6: 0.0
        joint_7: 0.0
        """
        assert len(joints_positions) == len(
            self.join_names), "Wrong number of joints, there should be "+str(len(self.join_names))
        joints_dict = dict(zip(self.join_names, joints_positions))

        return joints_dict

    def get_joints_position(self):
        self.gazebo.unpauseSim()
        #joint_positions = self.move_locobot_object.move_group_commander.get_current_joint_values()
        joint_positions = self.robot.arm.get_joint_angles()
        self.gazebo.pauseSim()
        return joint_positions

    def get_joints_velocity(self):
        self.gazebo.unpauseSim()
        joint_velocities = self.robot.arm.get_joint_velocities()
        self.gazebo.pauseSim()
        return joint_velocities

    def get_end_effector_pose(self):
        """
        Returns geometry_msgs/PoseStamped
            std_msgs/Header header
              uint32 seq
              time stamp
              string frame_id
            geometry_msgs/Pose pose
              geometry_msgs/Point position
                float64 x
                float64 y
                float64 z
              geometry_msgs/Quaternion orientation
                float64 x
                float64 y
                float64 z
                float64 w
        """
        self.gazebo.unpauseSim()

        gripper_pose = PoseStamped()
        gripper_pose.header.stamp = rospy.Time.now()
        gripper_pose.header.frame_id = 'base_link'
        #gripper_pose = self.move_locobot_object.ee_pose()
        ee_pose = self.robot.arm.get_ee_pose(self.robot.arm.configs.ARM.ARM_BASE_FRAME)
        cur_pos, cur_ori, cur_quat = ee_pose
        cur_pos = cur_pos.ravel()
        gripper_pose.pose.position.x = cur_pos[0]
        gripper_pose.pose.position.y = cur_pos[1]
        gripper_pose.pose.position.z = cur_pos[2]
        gripper_pose.pose.orientation.x = cur_quat[0]
        gripper_pose.pose.orientation.y = cur_quat[1]
        gripper_pose.pose.orientation.z = cur_quat[2]
        gripper_pose.pose.orientation.w = cur_quat[3]

        self.gazebo.pauseSim()
        return gripper_pose

    def get_end_effector_rpy(self):
        #gripper_rpy = self.move_locobot_object.ee_rpy()
        ee_pose = self.robot.arm.get_ee_pose('base_link')
        cur_pos, cur_ori, cur_quat = ee_pose
        quaternion = (cur_quat[0], cur_quat[1], cur_quat[2], cur_quat[3])
        euler = tf.transformations.euler_from_quaternion(quaternion)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        gripper_rpy = [roll, pitch, yaw]
        return gripper_rpy

    def wait_locobot_ready(self):
        """
        Function to wait the locobot move to its startup position
        """
        import time
        # Wait 20 seconds
        for i in range(5):
            print("WAITING ..." + str(i))
            sys.stdout.flush()
            time.sleep(1.0)

        print("WAITING...DONE")

    # Task Environment Methods
    # -------------------------------------

    def _init_env_variables(self):
        """
        Initial variables needed to be initialised
        each time we reset, at the start of an episode
        """
        raise NotImplementedError()

    def compute_reward(self, achieved_goal, goal, info):
        """
        Calculates the reward based on the observation given
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """
        Applies the given action to the simulation
        """
        raise NotImplementedError()

    def _get_obs(self):
        """
        Get observation from the simulation
        """
        raise NotImplementedError

    def _is_done(self, observation):
        """
        Checks if episode done based on the observation given
        """
        raise NotImplementedError()

# ------------------------------
