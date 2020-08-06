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
#from gym.envs.robotics import rotations, robot_env, utils
from pyrobot_gym.core import robot_mujoco_env, rotations, utils

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import String
from std_msgs.msg import Bool




class LocoBotMujocoEnv(robot_mujoco_env.RobotMujocoEnv):
    """
    Superclass for the LocoBot environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,n_actions,randomize_action_timesteps):
        """
        Initializes a new LocoBot environment

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
        # Initializes the SuperClass of RobotEnv
        super(LocoBotMujocoEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=5,
            initial_qpos=initial_qpos,randomize_action_timesteps=randomize_action_timesteps)

    # GoalEnv methods :: Define the Sparse/Binary or EuclideanDistance Reward
    # ----------------------------------

    # LocoBotMujocoEnv methods ::
    # -------------------------------
    def set_joints_position(self, action):
        result = utils.ctrl_set_action(self.sim, action)
        return result

    def close_gripper(self):
        self.sim.data.set_joint_qpos('joint_6', 0.)
        self.sim.data.set_joint_qpos('joint_7', 0.)
        self.sim.forward()

    def get_joints_position(self, sim):
        """Returns all joint positions and velocities associated with
        a robot.
        """
        if sim.data.qpos is not None and sim.model.joint_names:
            names = [n for n in sim.model.joint_names if n.startswith('joint_')]
            # DEBUG:
            #print('Joint 1 = {}'.format(sim.data.qpos[5]))
            #print('Joint 2 = {}'.format(sim.data.qpos[6]))
            #print('Joint 3 = {}'.format(sim.data.qpos[7]))
            #print('Joint 4 = {}'.format(sim.data.qpos[8]))
            #print('Joint 5 = {}'.format(sim.data.qpos[9]))
            joint_pos = np.array([sim.data.get_joint_qpos(name) for name in names])[:5]
            joint_vel = np.array([sim.data.get_joint_qvel(name) for name in names])[:5]
            return (
                joint_pos,
                joint_vel,
            )
        return np.zeros(0), np.zeros(0)

    def rest_back_to_original_state(self, original_joint_states):
        """Reset the MuJoCo Simulation back to State T"""
        self.sim.data.set_joint_qpos('joint_1', original_joint_states[0])
        self.sim.data.set_joint_qpos('joint_2', original_joint_states[1])
        self.sim.data.set_joint_qpos('joint_3', original_joint_states[2])
        self.sim.data.set_joint_qpos('joint_4', original_joint_states[3])
        self.sim.data.set_joint_qpos('joint_5', original_joint_states[4])
        self.sim.forward()
        
    """
    def _step_callback(self):
        self.sim.data.set_joint_qpos('joint_6', 0.)
        self.sim.data.set_joint_qpos('joint_7', 0.)
        self.sim.forward()
    """

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
        # DEBUG:
        #print('sim.data.site_xpos = {}'.format(self.sim.data.site_xpos))
        #print('sim.model.site_pos = {}'.format(self.sim.model.site_pos))
        #print('sites_offset = {}'.format(sites_offset))
        site_id = self.sim.model.site_name2id('target0')
        #self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.model.site_pos[site_id] = np.array([0.4049, 0.48, 0]) + self.goal
        #self.sim.model.site_pos[site_id] = self.goal
        #print('goal = {}'.format(self.goal))
        #print('sim.model.site_pos[site_id] = {}'.format(self.sim.model.site_pos[site_id]))

        self.sim.forward()


    def render(self, mode="human", width=500, height=500):
        return super(LocoBotMujocoEnv, self).render(mode, width, height)
