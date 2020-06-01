import gym
from gym import error, spaces
from gym.utils import seeding
from numbers import Number
from collections import OrderedDict
import os
import random
import numpy as np
import time
from timeit import default_timer as timer
from gym.envs.robotics import rotations, robot_env, utils

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import String
from std_msgs.msg import Bool

counter = 0

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def pyrobot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('joint_')]
        # DEBUG:
        #print(names)
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)

class PyRobotCoreEnv(robot_env.RobotEnv):
    """
    Superclass for all PyRobot environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type):
        """
        Initializes a new PyRobot environment

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        # Initializes the PyRobotEnv parameters
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = 'sparse'
        self.action_done_flag = False
        ##
        self.mode = "sim"
        #self._start_rospy()
        #if self.mode == "robot":
        #   self._start_rospy()
            #self.current_observation = None
        # Initializes the SuperClass of RobotEnv
        super(PyRobotCoreEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods :: Define the Sparse/Binary or EuclideanDistance Reward
    # ----------------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32) # Not achieved : -1 | Achieved : 0
        else: # Not binary/sparse reward, reward = -(goal_distance)
            return -d



    # PyRobotCoreEnv methods ::
    # -------------------------------
    def step(self, action):
        # Normalize the action within (-1,1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.mode == "sim":
            self._set_action(action)
            self.sim.step()
            self._step_callback()
            obs = self._get_obs()  # TODO: add real robot implementation

        elif self.mode == "robot":
            self._set_action(action)
            self._step_callback()
            obs = self._get_obs()

        done = False
        info = {
                'is_success': self._is_success(obs['achieved_goal'], self.goal)
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        print('[STEP] Rewards = {}'.format(reward))
        return obs, reward, done, info

    def reset(self):
        if self.mode == "sim":
            print('[RESET]')
            did_reset_sim = False
            while not did_reset_sim:
                did_reset_sim = self._reset_sim()
            self.goal = self._sample_goal().copy()
            obs = self._get_obs()
            return obs

        elif self.mode == "robot":
            rospy.loginfo('[DEBUG] Robot Reset')
            self.action_done_flag = False
            # Publish the "RESET" command to the PyRobot Client
            self.reset_publisher.publish(String("RESET"))
            start_time = timer()
            while(self.action_done_flag != True):
                self.action_done_flag = rospy.wait_for_message('/pyrobot/action_done', Bool).data
                # DEBUG:
                if self.action_done_flag == True:
                    rospy.loginfo('reset done = {}'.format(self.action_done_flag))
                else:
                    rospy.loginfo('Waiting Action to be Done ~')
                    end_time = timer()
                    print('time passed = {}'.format(end_time-start_time))
                    if end_time-start_time >= 5.:
                        self.action_done_flag = True
                        rospy.loginfo('Time Out and Pass ~~')


            self.goal = self._sample_goal().copy()
            obs = self._get_obs()
            print('Initial Joint Position = ')
            return obs

    def _step_callback(self):
        if self.block_gripper and self.mode == "sim":
            self.sim.data.set_joint_qpos('joint_6', 0.)
            self.sim.data.set_joint_qpos('joint_7', 0.)
            self.sim.forward()
        if self.block_gripper and self.mode == "robot":
            self.gripper_command_publisher.publish(String("OPEN"))

    def _set_action(self, action):
        """
        Define the Action: x,y,z relative position of the grippe + open/close gripper
        """
        assert action.shape == (4,) # define the action space = x,y,z,open/close
        action = action.copy()
        eff_pos_control, gripper_control = action[:3], action[3]  # 0.05 open | 0 close
        #print("gripper_control = {}".format(action[3]))
        #eff_pos_control *= 0.05  # limit the maximum change in position
        #print('eff_pos_control = {}'.format(eff_pos_control))
        #eff_rot_control = [0., 0., 0., 1]  # in this case, we fix the rotation of the end-effector
        eff_rot_control = [1., 0., 1., 0.]
        gripper_control = np.array([gripper_control, -gripper_control]) # l_finger(-1,0) r_finger(0,1)
        assert gripper_control.shape == (2,)
        # if the gripper is blocked, open the gripper
        if self.block_gripper:
            gripper_control = np.zeros_like(gripper_control)
        action = np.concatenate([eff_pos_control, eff_rot_control, gripper_control])
        self.action_done_flag = False
        if self.mode == "robot":
            action = action.copy()
            action = action.astype(np.float32)
            #global counter
            #counter = counter + 1
            print('Set Action displacement = {}'.format(action[:3]))
            self.action_publisher.publish(action)
            self.action_done_flag = rospy.wait_for_message('/pyrobot/action_done', Bool).data
            if self.action_done_flag == True:
                rospy.loginfo('[STATUS] Action Done')
            else:
                rospy.loginfo('[STATUS] Action not feasbile and passed')
            """
            while(self.action_done_flag != True):
                n = n + 1
                print('n = {}'.format(n))
                rospy.loginfo('action done = {}'.format(self.action_done_flag))
                #self.action_done_flag = rospy.wait_for_message('/pyrobot/action_done', Bool).data
                # DEBUG:
                if self.action_done_flag == True:
                    pass
                else:
                    rospy.loginfo('Waiting Action to be Done ~')
                    end_time = timer()
                    time_passed = end_time - start_time
                    print('time_passed = {}'.format(time_passed))
                    if time_passed >= 5.:
                        self.action_done_flag = True
                        rospy.loginfo('Time Out and Pass ~~')
            """
            global counter
            counter = counter + 1
            print('counter = {}'.format(counter))
            # DEBUG:
            #rospy.loginfo('[DEBUG] Action Published')
            #rospy.loginfo(action)
        elif self.mode == "sim":
            # Apply actions to simulation
            # DEBUG:
            #print('action = {}'.format(action))
            #utils.ctrl_set_action(self.sim, action)
            print('[Set Action] action = {}'.format(action))
            utils.mocap_set_action(self.sim, action)
            """
            robot_joints_pos, robot_joints_vel = pyrobot_get_obs(self.sim)
            action = robot_joints_pos[:5].astype(np.float32)
            self.action_publisher.publish(action)
            self.action_done_flag = rospy.wait_for_message('/pyrobot/action_done', Bool).data
            if self.action_done_flag == True:
                rospy.loginfo('[STATUS] Action Done')
            else:
                rospy.loginfo('[STATUS] Action not feasbile and passed')
                self.reset()
            """

    def _get_obs(self):
        """
        Get observation
        from pyrobot : (eff.x, eff.y, eff,z, joint1, ..., joint5,joint6, joint7, joint1_vel, ..., joint5_vel)
        """
        if self.mode == "robot":
            self.get_observation_command_publisher.publish(String("GET_OBS"))
            #if self.current_observation != None:
            #    prev_eff_pos = self.current_observation[:3]
            #else:
            #    prev_eff_pos = np.zeros((1,3))
            self.current_observation = np.array(rospy.wait_for_message(
                "/pyrobot/observation", numpy_msg(Floats)).data)
            eff_pos = self.current_observation[:3]
            #d_pos = eff_pos - prev_eff_pos
            #eff_vel = d_pos / self.sim.model.opt.timestep # TODO: ndim fix to 1
            eff_vel = np.zeros(3)
            robot_joints_pos = self.current_observation[3:10]
            robot_joints_vel = self.current_observation[10:]
            # TODO: CNN OBJECT DETECTION
            object_pos = object_rot = object_positional_vel = object_rotational_vel = object_relative_pos = np.zeros(0)
            gripper_state = robot_joints_pos[-2:]
            gripper_vel = np.zeros(2)

        elif self.mode == "sim":
            eff_pos = self.sim.data.get_site_xpos('robot0:end_effector')
            dt = self.sim.nsubsteps * self.sim.model.opt.timestep
            eff_vel = self.sim.data.get_site_xvelp('robot0:end_effector') * dt
            robot_joints_pos, robot_joints_vel = pyrobot_get_obs(self.sim)

            if self.has_object:
                object_pos = self.sim.data.get_site_xpos('object0')
                # object rotations in euler angle which is converted from a Rotation Matrix
                object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
                # velocities
                object_positional_vel = self.sim.data.get_site_xvelp('object0') * dt
                object_rotational_vel = self.sim.data.get_site_xvelr('object0') * dt
                # gripper state
                object_relative_pos = object_pos - eff_pos
                object_positional_vel = object_positional_vel - eff_vel  # obj vel also relative to gripper position
            else:
                object_pos = object_rot = object_positional_vel = object_rotational_vel = object_relative_pos = np.zeros(0)
            gripper_state = robot_joints_pos[-2:]
            gripper_vel = robot_joints_vel[-2:] * dt

        if not self.has_object:
            achieved_goal = eff_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        """
        Observation consists of
        - end effector absolute position
        - object absolute position
        - object relative positon to end effector
        - gripper state (open/close)
        - object rotation in euler angle
        - object positional velocity relative to end effoctor
        - object rotational velocity
        - end effoctor positional velocity
        - gripper velocity
        """
        obs = np.concatenate([
            eff_pos, object_pos.ravel(), object_relative_pos.ravel(), gripper_state, object_rot.ravel(),
            object_positional_vel.ravel(), object_rotational_vel.ravel(), eff_vel, gripper_vel
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }

    def _viewer_setup(self):
        """
        Setup the Camera view
        """
        body_id = self.sim.model.body_name2id('gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        """
        Visualize target
        """
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        """
        Reset the Simulation by reset all the joints to initial positions
        """
        self.sim.set_state(self.initial_state)
        #print('[RESET SIM] initial_state= {}'.format(self.initial_state))
        # Randomize initial position of the object
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]  #x,y
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                # object_initial pos = gipper pos + random-ness
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            # Assign the randomized position to the object
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()

        robot_joints_pos, robot_joints_vel = pyrobot_get_obs(self.sim)
        print('[Reset Sim] robot_initial_joint_pos = {}'.format(robot_joints_pos))

        return True

    def _sample_goal(self):
        """
        Initializes the goal position
        """
        if self.has_object:
            # Goal = {x,y,z}
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset  # z of object
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)  # add a random height of goal position
        else:
            # Always Relative to the gripper frame
            if self.mode == "sim":
                # DEBUG:
                #print('self.initial_gripper_xpos = {}'.format(self.initial_gripper_xpos[:3]))
                #goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                #goal = np.array([0.74780095, 0.60455716, 0.28715335])+ self.np_random.uniform(-self.target_range, self.target_range, size=3)
                goal = np.array([0.74780095, 0.40455716, 0.28715335]) + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                #goal = np.array([1.10934509, 0.74418712, 0.29003852])+ self.np_random.uniform(-self.target_range, self.target_range, size=3)

                #goal = np.array([1.23746173, 0.61563573, 0.32325541])
            elif self.mode == "robot":
                self.get_observation_command_publisher.publish(String("GET_OBS"))
                self.current_observation = np.array(rospy.wait_for_message(
                    "/pyrobot/observation", numpy_msg(Floats)).data)
                eff_pos = self.current_observation[:3]
                goal = eff_pos + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        print("============= Goal =================== {}".format(goal))
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        """
        Determine whether the desired goal is reached
        """
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        if self.mode == "robot":
            #self.reset_publisher.publish(String("RESET"))
            #self.get_observation_command_publisher.publish(String("GET_OBS"))
            #self.current_observation = np.array(rospy.wait_for_message(
            #    "/pyrobot/observation", numpy_msg(Floats)).data)
            #eff_pos = self.current_observation[:3]
            #self.initial_gripper_xpos = eff_pos
            # DEBUG:
            #rospy.loginfo('[DEBUG] env setup initial gripper xpos = {}'.format(self.initial_gripper_xpos))
            pass
        elif self.mode == "sim":
            # Setup the initial joint positions
            for name, value in initial_qpos.items():
                self.sim.data.set_joint_qpos(name, value)
            self.initial_state_2 = self.sim.get_state()
            utils.reset_mocap_welds(self.sim)
            self.sim.forward()

            # Move end effector into positions  | -0.498, 0.005, -0.431
            # + self.gripper_extra_height
            # DEBUG:
            #print('[ENV SETUP][BEFORE] initial_gripper_xpos = {}'.format(self.sim.data.get_site_xpos('robot0:end_effector')))
            #gripper_target = np.array([-0.498, 0.005, -0.431+ self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:end_effector')
            #gripper_target = np.array([-0.298, 0.005, -0.431])+ self.sim.data.get_site_xpos('robot0:end_effector')
            #gripper_target = self.sim.data.get_site_xpos('robot0:end_effector')
            #gripper_target = np.array([1.15360479, 0.74419554, 0.40428726])
            #gripper_target = np.array([1.10412612, 0.744193, 0.17162448])
            #gripper_rotation = np.array([1., 0., 1., 0.])
            #gripper_rotation = np.array([0., 0., 0., 1.])
            #print('[BEFORE] sim.data.mocap_pos = {}'.format(self.sim.data.mocap_pos))
            #print('gripper_target = {}'.format(gripper_target))
            #self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
            #self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
            #print('[After] sim.data.mocap_pos = {}'.format(self.sim.data.mocap_pos))
            #self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:end_effector').copy()
            #print('[ENV SETUP][AFTER] initial_gripper_xpos = {}'.format(self.initial_gripper_xpos))
            #for _ in range(10):
            #    self.sim.step()
            self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:end_effector').copy()
            print('[ENV SETUP][AFTER] initial_gripper_xpos = {}'.format(self.initial_gripper_xpos))
            print('[ENV SETUP][AFTER] sim.data.mocap_pos = {}'.format(self.sim.data.mocap_pos))

    def render(self, mode="human", width=500, height=500):
        if self.mode == "robot":
            pass
        else:
            return super(PyRobotCoreEnv, self).render(mode, width, height)

    """Functions for the Real Robot Implementation """
    def _start_rospy(self):
        # Indicate implementing on PyRobot/locobot
        #self.mode = "robot"
        self.rand_init = random.random()
        # Create a ROS Node for Publishing ans Subscribing Topics
        rospy.init_node("pyrobot_openai_gym_env")
        # Create the Reset Command Publisher
        self.reset_publisher = rospy.Publisher("/pyrobot/reset", String, queue_size=1)
        # Creatre the Position Updated Published
        self.position_updated_publisher = rospy.Publisher(
            "/pyrobot/received_position", String, queue_size=1)
        # Create the Action Publisher
        self.action_publisher = rospy.Publisher(
            "/pyrobot/action", numpy_msg(Floats), queue_size=100)
        # Create the Open Gripper Command Publisher
        self.gripper_command_publisher = rospy.Publisher("/pyrobot/gripper_command", String, queue_size=1)
        # Create the Get Observation Command Publisher
        self.get_observation_command_publisher = rospy.Publisher("/pyrobot/get_observation", String, queue_size=1)
        # Create the Observation Subscriber
        self.observation_subscriber = rospy.Subscriber(
            "/pyrobot/observation", numpy_msg(Floats), self._update_observation)
        # Create the Action Done Flag subscriber
        self.action_done_flag_subscriber = rospy.Subscriber(
            "/pyrobot/action_done", Bool, self._update_flag)
        rospy.sleep(2)
        r = rospy.Rate(10)  #10 hz
        rospy.loginfo('[DEBUG] ROS Node Initialized')
        #self.reset()
        return self

    def _update_flag(self, msg):
        self.action_done_flag = msg.data

    def _update_observation(self, msg):
        """
        Callback Function for Observation
        """
        self.current_observation = np.array(msg.data)
        # Publish the Position Updated Flag Status
        self.position_updated_publisher.publish('received')
