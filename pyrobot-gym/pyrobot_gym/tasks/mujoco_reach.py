import os
from gym import utils
from gym import error, spaces
import numpy as np
#from gym_pyrobot.envs.pyrobot_core_env import PyRobotCoreEnv
from pyrobot_gym.robots import LocoBotMujocoEnv
# Load the simulation environment XML
pwd = os.path.dirname(os.path.realpath(__file__))
#MODEL_XML_PATH = os.path.join(pwd, 'assets', 'locobot', 'reach.xml')
MODEL_XML_PATH = os.path.abspath(os.path.join(pwd, os.pardir, 'assets', 'tasks', 'reach.xml'))
# DEBUG:
print("path = {}".format(MODEL_XML_PATH))

"""
Boundaries of the Configuration Space
"""
BOUNDS_CEILLING = .45
BOUNDS_FLOOR = .15
BOUNDS_LEFTWALL = .45
BOUNDS_RIGHTWALL = -.45
BOUNDS_FRONTWALL = .5
BOUNDS_BACKWALL = -.13


class LocoBotMujocoReachEnv(LocoBotMujocoEnv, utils.EzPickle):
    def __init__(self,
                 reward_type='sparse',
                 n_actions=4,
                 has_object=False,
                 block_gripper=True,
                 n_substeps=20,
                 gripper_extra_height=0.2,
                 target_in_the_air=True,
                 target_offset=0.0,
                 obj_range=0.15,
                 target_range=0.15,
                 distance_threshold=0.05):
        print("[LocoBotMujocoReachEnv] START init LocoBotMujocoReachEnv")
        # Load as the Environment Parameters
        self.get_params(reward_type, n_actions, has_object, block_gripper, n_substeps, gripper_extra_height,
                        target_in_the_air, target_offset, obj_range, target_range, distance_threshold)
        # Set the Environment Parameters
        LocoBotMujocoEnv.__init__(
            self,MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.25,
            distance_threshold=0.05,
            initial_qpos=self.initial_qpos,
            reward_type=self.reward_type,
            n_actions=self.n_actions)
        utils.EzPickle.__init__(self)

        # Setup the Observation Space and Action Space here
        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def get_params(self,reward_type, n_actions, has_object, block_gripper, n_substeps, gripper_extra_height,
                    target_in_the_air, target_offset, obj_range, target_range, distance_threshold):
        self.initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'joint_1': -1.22487753e-05,
            'joint_2': 6.71766300e-03,
            'joint_3': 7.30874027e-03,
            'joint_4': 3.80183559e-03,
            'joint_5': -5.06792684e-05
        }
        # initial goal = startup end effector position
        self.initial_gripper_xpos = np.array([4.11812983e-01, 9.47465341e-05, 4.04977648e-01])
        self.goal = self.initial_gripper_xpos
        self.n_actions = n_actions
        self.reward_type = reward_type
        self.has_object = has_object
        self.block_gripper = block_gripper
        self.n_substeps = n_substeps
        self.gripper_extra_height = gripper_extra_height
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.use_random_goal = False
        self.last_joint_positions = None
        self.last_gripper_xpos = None
        self.valid_move = True

    def _set_init_pose(self):
        """
        1.Sets the Robot to its startup initial position whenever RESET
        2. Sample a goal
        """
        # Setup the initial [x,y,z]position and joint positions
        for name, value in self.initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()
        robot_pos = np.array([0.4049, 0.48, 0])
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:end_effector').copy() - robot_pos
        print('[RESET] initial_gripper_xpos = {}'.format(self.initial_gripper_xpos))

        # Sample a goal
        if self.use_random_goal:
            print("SAMPLING a new DESIRED GOAL Position")
            self.goal = self._sample_goal(self.initial_gripper_xpos)
        else:
            self.goal = np.array([0.32126746, -0.03747156, 0.26531804])

        # Record the joint pos and end effector pos right after the reset
        self.last_gripper_xpos = self.sim.data.get_site_xpos('robot0:end_effector')
        self.last_joint_positions, _ = self.get_joints_position(self.sim)
        print('[RESET] Initial Joint Position = {}'.format(self.last_joint_positions))

    def _set_action(self, action):
        """
        Define the Action: 5 Joint Positions + (open/close gripper)
        """
        assert action.shape == (4,)
        action = action.copy()
        action = action*0.25 # restrict the action
        print('[Set Action] Action = {}'.format(action))
        self.valid_move = self.set_joints_position(action)

    def _get_obs(self):
        """
        Observation consists of
        - end effector absolute position
        - relative displacement from end-effoctor to desired goal position
        - joint positions
        """
        robot_pos = np.array([0.4049, 0.48, 0])
        eff_pos = self.sim.data.get_site_xpos('robot0:end_effector') - robot_pos
        relative_dis_ee_goal = np.array([self.goal_distance(self.goal, eff_pos)])
        joint_positions, _ = self.get_joints_position(self.sim)
        print('[Get Obs] effpos = {}'.format(eff_pos))
        print('[Get Obs] goal = {}'.format(self.goal))
        print('[Get Obs] distance = {}'.format(relative_dis_ee_goal))
        print('[Get Obs] joint_pos = {}'.format(joint_positions))
        self.last_joint_positions = joint_positions
        obs = np.concatenate([eff_pos,
                              relative_dis_ee_goal,
                              joint_positions])
        return {'observation': obs.copy(),
                    'achieved_goal': eff_pos.copy(),
                    'desired_goal': self.goal.copy()}


    def _is_success(self, achieved_goal, desired_goal):
        """
        Determine whether the desired goal is reached
        Reached = 1
        Not Reached / Action Not Feasible = 0
        """
        if self.valid_move:
            d = self.goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)
        else:
            print("Action is not feasible! Reward")
            return np.float32(0.0)


    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        if achieved_goal.size != 3:  # if calculating the batch reward
            self.valid_move = True
        d = self.goal_distance(achieved_goal, goal)
        if self.valid_move:
            if self.reward_type == "sparse":
                return -(d > self.distance_threshold).astype(np.float32) # Not achieved : -1 | Achieved : 0
            else: # Not binary/sparse reward, reward = -(goal_distance)
                return -d
        else:
            return np.float32(-1.0)

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _sample_goal(self, initial_ee_pos):

        sample_goal = True
        while sample_goal == True:
            #goal = np.array([0.74780095, 0.40455716, 0.28715335]) + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal = initial_ee_pos + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            conditions = [goal[0] <= BOUNDS_FRONTWALL, goal[0] >= BOUNDS_BACKWALL,
                          goal[1] <= BOUNDS_LEFTWALL, goal[1] >= BOUNDS_RIGHTWALL,
                          goal[2] <= BOUNDS_CEILLING, goal[2] >= BOUNDS_FLOOR]
            violated_boundary = False
            for condition in conditions:
                if not condition:
                    violated_boundary = True
                    break
            if violated_boundary == True:
                print('[Sample Goal] Sampled Goal Exists Boundaries --> Need to resample a valid goal')
                sample_goal = True
            else:
                sample_goal = False
        # end loop
        print('[Sample Goal] Sampled Goal = {}'.format(goal))
        return goal.copy()
