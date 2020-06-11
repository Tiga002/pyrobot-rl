import os
from gym import utils
import copy
import rospy
from gym import spaces
from pyrobot_gym.robots import locobot_gazebo_env
from gym.envs.registration import register
import numpy as np
from sensor_msgs.msg import JointState
from pyrobot_gym.core.openai_ros_common import ROSLauncher
from pyrobot_gym.tasks.task_commons import LoadYamlFileParamsTest

"""
Boundaries of the Configuration Space
"""
BOUNDS_CEILLING = .45
BOUNDS_FLOOR = .15
BOUNDS_LEFTWALL = .45
BOUNDS_RIGHTWALL = -.45
BOUNDS_FRONTWALL = .5
BOUNDS_BACKWALL = -.13

class LocoBotGazeboReachEnv(locobot_gazebo_env.LocoBotGazeboEnv, utils.EzPickle):
    def __init__(self):
        # Launch the Task Simulation Environment

        # 1: Load Params from the config YAML file to this Task Environment
        LoadYamlFileParamsTest(rospackage_name="pyrobot_rl",
                               rel_path_from_package_to_file="pyrobot-gym/pyrobot_gym/tasks/config",
                               yaml_file_name="locobot_gazebo_reach.yaml")

        # 2: get the ROS workspace abs path from the launch file
        ros_ws_abspath = rospy.get_param("/locobot/ros_ws_abspath", None)
        ros_ws_src_path = rospy.get_param("/locobot/ros_ws_src_path", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        # Roslaunch the simulation world(task) environment
        #ROSLauncher(rospackage_name="locobot_gazebo",
        #            #launch_file_name="start_world.launch",
        #            launch_file_name="gazebo.launch",
        #            ros_ws_abspath=ros_ws_abspath)

        rospy.logdebug("Entered LocoBotGazeboReachEnv")
        self.get_params()

        super(LocoBotGazeboReachEnv, self).__init__(ros_ws_abspath, ros_ws_src_path)

        # Define the action space here (since task specific)
        #self.action_space = spaces.Discrete(self.n_actions)
        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        super(LocoBotGazeboReachEnv, self).__init__(ros_ws_abspath, ros_ws_src_path)

    def get_params(self):
        """
        Get Parms from the YAML File
        """
        self.n_actions = rospy.get_param('/locobot/n_actions')
        self.n_observations = rospy.get_param('/locobot/n_observations')
        self.n_max_iterations = rospy.get_param('/locobot/n_max_iterations')
        self.threshold_error = rospy.get_param('/locobot/threshold_error')

        self.initial_joint_pos = rospy.get_param('/locobot/initial_joint_pos')
        self.initial_ee_pos = rospy.get_param('/locobot/initial_ee_pos')
        self.goal_ee_pos = rospy.get_param('/locobot/goal_ee_pos')

        self.position_delta = rospy.get_param('/locobot/position_delta')
        self.step_punishment = rospy.get_param('/locobot/step_punishment')
        #self.closer_reward = rospy.get_param('/locobot/closer_reward')
        #self.impossible_movement_punishement = rospy.get_param(
        #    '/locobot/impossible_movement_punishement')
        self.reached_goal_reward = rospy.get_param(
            '/locobot/reached_goal_reward')

        self.max_distance_from_ee_to_goal = rospy.get_param('/locobot/max_distance_from_ee_to_goal')
        self.threshold_error = rospy.get_param('/locobot/threshold_error')
        self.use_random_goal = rospy.get_param('/locobot/use_random_goal')
        self.random_target_range = rospy.get_param('/locobot/random_target_range')

        # Goal always in python list
        self.desired_position = [self.goal_ee_pos["x"],
                                 self.goal_ee_pos["y"],
                                 self.goal_ee_pos["z"]]
        #self.gripper_rotation = [1., 0., 1., 0.]
        self.gripper_rotation = [0.245, 0.613, -0.202, 0.723]
        self.last_joint_positions = None
        self.last_gripper_target = None



    def _set_init_pose(self):
        """
        Sets the Robot to its startup initial position whenever RESET
        Simulation will be unpaused for this purpose
        """
        self.reached_goal_reward = rospy.get_param(
            '/locobot/reached_goal_reward')

        self.max_distance_from_ee_to_goal = rospy.get_param('/locobot/max_distance_from_ee_to_goal')

        # DEBUG: Check the initial joint positions
        rospy.logdebug("Initial Joint Positions:")
        rospy.logdebug(self.initial_joint_pos)

        self.movement_result = self._set_startup_position()
        gripper_pose = self.get_end_effector_pose()
        self.last_gripper_target = [gripper_pose.pose.position.x,
                                    gripper_pose.pose.position.y,
                                    gripper_pose.pose.position.z]
        rospy.logdebug("Initial EE Position = {}".format(self.last_gripper_target))
        self.last_joint_positions = self.get_joints_position()
        rospy.logdebug("Initial Joint Position = {}".format(self.last_joint_positions))

        # Samle a Goal or Not
        if self.use_random_goal:
            rospy.logdebug("SAMPLING a new DESIRED GOAL Position")
            self.desired_position = self._sample_goal(self.last_gripper_target)


        self.last_action= "INIT"
        rospy.logdebug("Set to Startup initial pose ---> " + str(self.movement_result))

    def _init_env_variables(self):
        """
        Initial variables needed to be initialised everytime we reset at the start
        of an episode.
        The simulation will be paused, therefore all the data retrieved has to be
        from a system that doesnt need the simulation running, like variables where the
        callbackas have stored last know sesnor data.
        :return:
        """
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("Init Env Variables...END")

    def _set_action(self, action):

        assert action.shape == (4,)
        action = action*0.25 # restrict the action
        cur_joint_pos = self.last_joint_positions
        cur_joint_pos[0] += action[0]
        cur_joint_pos[1] += action[1]
        cur_joint_pos[2] += action[2]
        cur_joint_pos[3] += action[3]
        cur_joint_pos[4] = cur_joint_pos[4]

        # Clip the joint angle
        cur_joint_pos = np.array(cur_joint_pos)
        for i in range(5):
            cur_joint_pos[i] = np.clip(cur_joint_pos[i], -1.25, 1.25)
        # Apply action to simulation
        self.movement_result = self.set_trajectory_joints(cur_joint_pos)
        if self.movement_result:
            rospy.loginfo("Joint Positions are feasbile ... " + str(cur_joint_pos))
        else:
            rospy.logerr("Joint Positions are not feasbile ... " + str(cur_joint_pos))


    def _get_obs(self):
        """
        Observation consists of
        - end effector absolute position
        - relative displacement from end-effoctor to desired goal position
        - joint positions
        """
        # End effector absolute position
        end_effector_pose = self.get_end_effector_pose()
        end_effector_position = np.array([end_effector_pose.pose.position.x,
                                          end_effector_pose.pose.position.y,
                                          end_effector_pose.pose.position.z])
        # Relative displacement from end-effoctor to desired goal position
        relative_dis_ee_goal = np.array([self.calculate_distance_between(
                                self.desired_position,
                                end_effector_position)])
        # Joint positions
        joint_positions = self.get_joints_position()
        self.last_joint_positions = joint_positions
        obs = np.concatenate([end_effector_position,
                            relative_dis_ee_goal,
                            joint_positions])
        rospy.loginfo('[Get OBS] joint_positions = {}'.format(joint_positions))
        rospy.loginfo('[Get OBS] eff_pos = {}'.format(end_effector_position))
        rospy.loginfo('[Get OBS] goal = {}'.format(self.desired_position))
        obs_dict = {'observation': obs.copy(),
                    'achieved_goal': end_effector_position.copy(),
                    'desired_goal': np.array(self.desired_position)}
        #rospy.logdebug("Observations =======>>> {}".format(obs_dict))

        return obs_dict

    def _is_done(self, observations):
        """
        If the last Action didnt succeed, it means that tha position asked was imposible therefore the episode must end.
        It will also end if it reaches its goal.
        """
        current_eff_pos = observations['observation'][:3].tolist()

        done = self.check_if_done(
            self.movement_result, self.desired_position, current_eff_pos, self.threshold_error)
        return done

    def compute_reward(self, achieved_goal, goal, info):
        """
        We punish each step that it passes
            - without achieveing the goal within a threshold distance [-1], or
            - action is a position that is impoosible to reach [-1]
        Otherwise, we reward the agent
            - If the end_effector gets to a position near the goal within the threshold distance [0]
        """
        if achieved_goal.size == 3:
            achieved_goal = achieved_goal.tolist()
            goal = goal.tolist()
            reward = self.calculate_step_reward(self.movement_result,
                                                goal,
                                                achieved_goal,
                                                self.threshold_error)
            rospy.logdebug(">>>> Step Reward >>>>" + str(reward))
        else:
            reward = self.calculate_batch_reward(achieved_goal, goal, self.threshold_error)
            rospy.logdebug(" -----Batch Reward = {}".format(reward))

        return reward

    def check_if_done(self, movement_result, desired_position, current_eff_pos, threshold_error=1e-02):
        """
        It check the agent has finished the task or not
        """
        done = np.float32(0.0)

        if movement_result:
            distance = self.calculate_distance_between(desired_position, current_eff_pos)
            rospy.logerr('distance = {}'.format(distance))
            if distance < abs(threshold_error):
                done = np.float32(1.0) # True
                rospy.logdebug("Reach the desired position")
            else:
                done = np.float32(0.0) # False
                rospy.logdebug("Still Reaching the desired position")
        else:
            # movement_result = false, end the episode
            done = np.float32(0.0) # False
            rospy.logdebug("The action position is not possible --> end the episode")

        return done

    def calculate_batch_reward(self, achieved_goal, goal, threshold_error=1e-02):
        achieved_goal = achieved_goal.tolist()
        goal = goal.tolist()
        distance = self.calculate_distance_between(achieved_goal, goal)
        return -(distance > threshold_error).astype(np.float32)

    def calculate_step_reward(self, movement_result, desired_position, current_eff_pos, threshold_error=1e-02):
        """
        Calculate how much reward the agent can get through this step
        """
        if movement_result:
            distance = self.calculate_distance_between(desired_position, current_eff_pos)
            rospy.logerr('distance = {}'.format(distance))
            if distance < threshold_error:
                reward = np.float32(self.reached_goal_reward) # 0
                rospy.logdebug("Reached the desired position! Reward = {}".format(reward))
            else:
                reward = np.float32(self.step_punishment) # -1
                rospy.logdebug("Not yet reaching the desired position ~ Reward = {}".format(reward))
        else:
            reward = np.float32(self.step_punishment) # -1
            rospy.logdebug("Action is not feasible! Reward = {}".format(reward))

        return reward

    def calculate_distance_between(self, pos_1, pos_2):
        """
        Calculate the Euclidian distance between 2 position vectors in [python list]
        """
        distance = np.linalg.norm(np.array(pos_1) - np.array(pos_2), axis=-1)
        return distance

    def _sample_goal(self, initial_ee_pos):
        """
        Return Goal in python list
        """
        end_effector_starting_pos = np.array(initial_ee_pos)
        sample_goal = True
        while sample_goal == True:
            goal = end_effector_starting_pos + self.np_random.uniform(
                                                -0.1,
                                                self.random_target_range,
                                                size=3)
            conditions = [goal[0] <= BOUNDS_FRONTWALL, goal[0] >= BOUNDS_BACKWALL,
                          goal[1] <= BOUNDS_LEFTWALL, goal[1] >= BOUNDS_RIGHTWALL,
                          goal[2] <= BOUNDS_CEILLING, goal[2] >= BOUNDS_FLOOR]
            violated_boundary = False
            for condition in conditions:
                if not condition:
                    violated_boundary = True
                    break
            if violated_boundary == True:
                rospy.logerr('Need to resample a valid goal')
                sample_goal = True
            else:
                sample_goal = False

        rospy.loginfo("Sampled Goal = {}".format(goal))
        goal = goal.tolist()
        return goal
