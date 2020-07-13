import gym
import numpy as np
import random

import pyrobot_gym

class RandomizedEnvironment:
    """
    Class responsible for randomize the environment parameters
    """
    def __init__(self, experiment, parameter_ranges, goal_range):
        self._experiment = experiment
        self._parameter_ranges = parameter_ranges
        self._goal_range = goal_range
        self._params = [0]
        random.seed(123)

    def sample_env(self):
        min = self._parameter_ranges[0]
        max = self._parameter_ranges[1]
        pick = min + (max-min)*random.random()

        self._params = np.array([pick])
        print('Sample an environment ....')
        self._env = gym.make(self._experiment)
        self._env.reward_type = "sparse"
        # Mass of each link
        # Shoulder link
        shoulder_link_mass = self._env.get_property('shoulder_link', 'body_mass')
        self._env.set_property('shoulder_link', 'body_mass', shoulder_link_mass*random.uniform(0.25,1.5))
        # Elbow link
        elbow_link_mass = self._env.get_property('elbow_link', 'body_mass')
        self._env.set_property('elbow_link', 'body_mass', elbow_link_mass*random.uniform(0.25,1.5))
        # Forearm link
        forearm_link_mass = self._env.get_property('forearm_link', 'body_mass')
        self._env.set_property('forearm_link', 'body_mass', forearm_link_mass*random.uniform(0.25,1.5))
        # Wrist link
        wrist_link_mass = self._env.get_property('wrist_link', 'body_mass')
        self._env.set_property('wrist_link', 'body_mass', wrist_link_mass*random.uniform(0.25,1.5))
        # Gripper link
        gripper_link_mass = self._env.get_property('gripper_link', 'body_mass')
        self._env.set_property('gripper_link', 'body_mass', gripper_link_mass*random.uniform(0.25,1.5))
        # Finger R link
        finger_r_mass = self._env.get_property('finger_r', 'body_mass')
        self._env.set_property('finger_r', 'body_mass', finger_r_mass*random.uniform(0.25,1.5))
        # Finger L link
        finger_l_mass = self._env.get_property('finger_l', 'body_mass')
        self._env.set_property('finger_l', 'body_mass', finger_l_mass*random.uniform(0.25,1.5))

        # Joint Damping
        joint_1_damping = self._env.get_property('joint_1', 'dof_damping')
        self._env.set_property('joint_1', 'dof_damping', joint_1_damping*random.uniform(0.2,20))
        joint_2_damping = self._env.get_property('joint_2', 'dof_damping')
        self._env.set_property('joint_2', 'dof_damping', joint_2_damping*random.uniform(0.2,20))
        joint_3_damping = self._env.get_property('joint_3', 'dof_damping')
        self._env.set_property('joint_3', 'dof_damping', joint_3_damping*random.uniform(0.2,20))
        joint_4_damping = self._env.get_property('joint_4', 'dof_damping')
        self._env.set_property('joint_4', 'dof_damping', joint_4_damping*random.uniform(0.2,20))
        joint_5_damping = self._env.get_property('joint_5', 'dof_damping')
        self._env.set_property('joint_5', 'dof_damping', joint_5_damping*random.uniform(0.2,20))

        # Object Mass
        self._env.set_property('object0', 'body_mass', random.uniform(0.1,0.4))

        # Controller Gains
        joint_1_gain = self._env.get_property('arm:joint_1', 'actuator_gainprm')
        joint_1_gain[0] = joint_1_gain[0]*random.uniform(0.5,2)
        self._env.set_property('arm:joint_1', 'actuator_gainprm', joint_1_gain)

        joint_2_gain = self._env.get_property('arm:joint_2', 'actuator_gainprm')
        joint_2_gain[0] = joint_2_gain[0]*random.uniform(0.5,2)
        self._env.set_property('arm:joint_2', 'actuator_gainprm', joint_2_gain)

        joint_3_gain = self._env.get_property('arm:joint_3', 'actuator_gainprm')
        joint_3_gain[0] = joint_3_gain[0]*random.uniform(0.5,2)
        self._env.set_property('arm:joint_3', 'actuator_gainprm', joint_3_gain)

        joint_4_gain = self._env.get_property('arm:joint_4', 'actuator_gainprm')
        joint_4_gain[0] = joint_4_gain[0]*random.uniform(0.5,2)
        self._env.set_property('arm:joint_4', 'actuator_gainprm', joint_4_gain)

    def get_env(self):
        return self._env, self._params

    def close_env(self):
        self._env.close()

    def get_goal(self):
        return
