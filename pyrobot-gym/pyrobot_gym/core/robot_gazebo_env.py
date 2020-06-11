import rospy
import time
import gym
from gym.utils import seeding
from .gazebo_connection import GazeboConnection
from .controllers_connection import ControllersConnection

#from pyrobot_gym.msg import RLExperimentInfo

class RobotGazeboEnv(gym.GoalEnv):

    def __init__(self, robot_name_space, controllers_list, reset_controls,
                start_init_physics_parameters=True,
                reset_world_or_sim="SIMULATION"):
        # To reset simulations
        rospy.logdebug("START init RobotGazeboEnv")
        # Create the Gazebo Connection Instance
        self.gazebo = GazeboConnection(start_init_physics_parameters, reset_world_or_sim)
        self.controllers_object = ControllersConnection(namespace=robot_name_space, controllers_list=controllers_list)
        self.reset_controls = reset_controls
        self.seed()

        # Setup ROS related variables
        self.episode_num = 0
        self.cumulated_episode_reward = 0
        #self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)

        # We Unpause the simulation and reset the controllers if needed
        """
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        """
        self.gazebo.unpauseSim()
        if self.reset_controls:
            self.controllers_object.reset_controllers()



        rospy.logdebug("END init RobotGazeboEnv")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done, info
        """

        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """
        rospy.loginfo("START STEP OpenAI ROS")

        self.gazebo.unpauseSim()
        self._set_action(action)
        self.gazebo.pauseSim()
        obs = self._get_obs()
        #done = self._is_done(obs)
        #done = False
        info = {'is_success': self._is_done(obs)}
        # only in --play
        if info['is_success'] == 1:
            done = True
            time.sleep(1)
            rospy.logwarn("=======  DONE ==========")
        else:
            done = False

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        print('[STEP] Rewards = {}'.format(reward))
        print('[STEP] info = {}'.format(info))
        self.cumulated_episode_reward += reward

        rospy.loginfo("END STEP OpenAIROS===================================")
        return obs, reward, done, info

    def reset(self):
        rospy.loginfo("Reseting RobotGazeboEnv")
        self._reset_sim()
        self._init_env_variables()
        self._update_episode()
        obs = self._get_obs()
        rospy.loginfo("END Reseting RobotGazeboEnv")
        return obs

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

    def _update_episode(self):
        """
        Pubishes the cumulated reward of the episode and
        increases the episode number by one.
        :return
        """
        rospy.logdebug("PUBLISHING REWARD ...")
        #self._publish_reward_topic(self.cumulated_episode_reward, self.episode_num)
        rospy.logdebug("PUBLISHING REWARD ... DONE ="
                        + str(self.cumulated_episode_reward)
                        + ", EP = " + str(self.episode_num))

        self.episode_num += 1
        self.cumulated_episode_reward = 0

    """
    def _publish_reward_topic(self, reward, episode_num=1):

        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:

        reward_msg = RLExperimentInfo()
        reward_msg.episode_num = episode_num
        reward_msg.episode_reward = reward
        # Publish the Reward MSG hereeee
        self.reward_pub.publish(reward_msg)
    """
    # Extension methods
    # ---------------------

    def _reset_sim(self):
        """
        Resets a simulatio
        """
        rospy.logdebug("RESET SIM STARTS")
        if self.reset_controls:
            rospy.logdebug("RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()
        else:
            rospy.logwarn("DONT RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        rospy.logdebug("RESET SIM END")
        return True

    def _set_init_pose(self):
        """
        Sets the Robot in its initial pose
        """
        raise NotImplementedError()

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation
        systems are operational
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """Indicates whether or not the episode is done ( the robot has fallen for example).
        """
        raise NotImplementedError()

    def compute_reward(self, achieved_goal, goal, info):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        raise NotImplementedError()

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
