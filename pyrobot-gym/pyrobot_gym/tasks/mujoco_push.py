import os
from gym import utils
from gym import error, spaces
import numpy as np
from pyrobot_gym.robots import LocoBotMujocoEnv
from pyrobot_gym.core import rotations

# Load the simulation environment XML
pwd = os.path.dirname(os.path.realpath(__file__))
#MODEL_XML_PATH = os.path.join(pwd, 'assets', 'locobot', 'reach.xml')
MODEL_XML_PATH = os.path.abspath(os.path.join(pwd, os.pardir, 'assets', 'tasks', 'push.xml'))
# DEBUG:
print("path = {}".format(MODEL_XML_PATH))

"""
Boundaries of the Configuration Space
"""
GOAL_LEFTWALL = .35
GOAL_RIGHTWALL = -.35
GOAL_FRONTWALL = .35
GOAL_BACKWALL = .25

class LocoBotMujocoPushEnv(LocoBotMujocoEnv, utils.EzPickle):
    def __init__(self,
                 reward_type='sparse',
                 n_actions=4,
                 has_object=True,
                 block_gripper=True,
                 n_substeps=20,
                 gripper_extra_height=0.0,
                 target_in_the_air=False,
                 target_offset=0.0,
                 obj_range=0.2,
                 target_range=0.15,
                 distance_threshold=0.05):
        print("[LocoBotMujocoReachEnv] START init LocoBotMujocoReachEnv")
        # Load as the Environment Parameters
        self.get_params(reward_type, n_actions, has_object, block_gripper, n_substeps, gripper_extra_height,
                    target_in_the_air, target_offset, obj_range, target_range, distance_threshold)
        # Set the Environment Parameters
        LocoBotMujocoEnv.__init__(
            self,MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.2,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=self.initial_qpos,
            reward_type=self.reward_type,
            n_actions=self.n_actions,
            randomize_action_timesteps=True)
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
            'joint_5': -5.06792684e-05,
            'object0:joint': [1.25, 0.53, 0.025, 1., 0., 0., 0.],
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
        self.use_random_goal = True
        self.last_joint_positions = None
        self.last_gripper_xpos = None
        self.last_object_position = None
        self.valid_move = True
        self.height_offset = 0.0
        self.randomize_action_timesteps=True

    def _set_init_pose(self):
        """
        1.Sets the Robot to its startup initial position whenever RESET
        2. Randomize starting position of the object
        3. Sample a goal
        """
        # Setup the initial [x,y,z]position and joint positions
        for name, value in self.initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()
        robot_pos = self.sim.data.get_site_xpos('robot0:base')
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:end_effector').copy() - robot_pos
        print('[RESET] initial_gripper_xpos = {}'.format(self.initial_gripper_xpos))

        # Randomize the starting position of the object
        object_xpos = self.initial_gripper_xpos[:2]
        sample_position = True
        while sample_position == True:
            object_xpos  = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            conditions = [object_xpos[0] <= GOAL_FRONTWALL, object_xpos[0] >= GOAL_BACKWALL,
                          object_xpos[1] <= GOAL_LEFTWALL, object_xpos[1] >= GOAL_RIGHTWALL]
            violated_boundary = False
            for condition in conditions:
                if not condition:
                    violated_boundary = True
                    break
            if violated_boundary == True:
                print('[Sample Object Position] Object Position exceeds Configuration space --> Need to resample')
                sample_position = True
            elif np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                print('[Sample Object Position] Need to resample')
                sample_position = True
            else:
                sample_position = False
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        #object_xpos = np.array([0.33662441, -0.11767662])
        object_qpos[:2] = object_xpos + robot_pos[:2]
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()
        self.last_object_position = self.sim.data.get_site_xpos('object0') - robot_pos
        print('[RESET] Object Position = {}'.format(self.last_object_position))

        # Sample a goal
        if self.use_random_goal:
            print("SAMPLING a new DESIRED GOAL Position")
            self.goal = self._sample_goal(self.last_object_position)
        else:
            self.goal = np.array([0.34242378, -0.19046866, 0.])

        #Record the joint pos and end effector pos right after the reset
        self.last_gripper_xpos = self.sim.data.get_site_xpos('robot0:end_effector').copy()- robot_pos
        self.last_joint_positions, _ = self.get_joints_position(self.sim)
        print('[RESET] Initial Joint Position = {}'.format(self.last_joint_positions))



    def _set_action(self, action):
        """
        Define the Action: 5 Joint Positions + (open/close gripper)
        """
        #assert action.shape == (4,)
        action = action.copy()
        action = action*0.25 # restrict the action
        print('[Set Action] Action = {}'.format(action))
        self.valid_move = self.set_joints_position(action)
        self.sim.data.set_joint_qpos('joint_6', 0)
        self.sim.data.set_joint_qpos('joint_7', 0)
        self.sim.forward()


    def _get_obs(self):
        """
        Observation consists of
        - end effector absolute position
        - relative displacement from end-effoctor to desired goal position
        - joint positions
        """
        # dt
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # Robot position offset
        #robot_pos = np.array([0.4049, 0.48, 0])
        robot_pos = self.sim.data.get_site_xpos('robot0:base')
        # End Effector position relative to the robot's base frame
        eff_pos = self.sim.data.get_site_xpos('robot0:end_effector') - robot_pos
        # End Effector Positional Velocity
        eff_velp = self.sim.data.get_site_xvelp('robot0:end_effector') * dt
        #relative_dis_ee_goal = np.array([self.goal_distance(self.goal, eff_pos)])
        # Joints Position and Velocities
        joint_positions, joint_velocities = self.get_joints_position(self.sim)

        # Target Object
        # Object Position relative to the robot's base frame
        object_pos = self.sim.data.get_site_xpos('object0') - robot_pos
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        # Object position&velocity relative to the gripper
        object_rel_pos = object_pos - eff_pos
        object_rel_velp = object_velp - eff_velp

        # DEBUG:
        print('base_location = {}'.format(robot_pos))
        print('[Get Obs] eff_pos = {}'.format(eff_pos))
        print('[Get Obs] eff_velp = {}'.format(eff_velp))
        print('[Get Obs] goal = {}'.format(self.goal))
        print('[Get Obs] joint_pos = {}'.format(joint_positions))
        print('[Get Obs] object_pos = {}'.format(object_pos))
        self.last_joint_positions = joint_positions
        obs = np.concatenate([eff_pos,
                              eff_velp,
                              joint_positions,
                              joint_velocities,
                              object_pos.ravel(),
                              object_rel_pos.ravel(),
                              object_rel_velp.ravel(),
                              object_rot.ravel(),
                              object_velr.ravel()])

        # Current object position
        achieved_goal = np.squeeze(object_pos.copy())

        return {'observation': obs.copy(),
                'achieved_goal': achieved_goal.copy(),
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
            print("[Is Success] Action is not feasible")
            return np.float(0.0)

    def compute_reward(self, achieved_goal, goal, info):
        d = self.goal_distance(achieved_goal, goal)
        if self.valid_move:
            if self.reward_type == "sparse":
                return -(d > self.distance_threshold).astype(np.float32) # Not achieved : -1 | Achieved : 0
            else:
                return -d
        else:
            return np.float32(-1.0)  # Reward = -1 if the action is not valid

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _sample_goal(self, object_position):
        """
        Goal Position of the object being pushed
        """
        sample_goal = True
        while sample_goal == True:
            goal = object_position + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset  # Object goal is on the ground
            conditions = [goal[0] <= GOAL_FRONTWALL, goal[0] >= GOAL_BACKWALL,
                          goal[1] <= GOAL_LEFTWALL, goal[1] >= GOAL_RIGHTWALL]
            violated_boundary = False
            for condition in conditions:
                if not condition:
                    violated_boundary = True
                    break
            if violated_boundary == True:
                sample_goal = True
            elif np.linalg.norm(object_position[:2] - goal[:2]) < 0.05:
                sample_goal = True
            else:
                sample_goal = False
            # end loop
        print('[Sample Goal] Sampled Goal Position for the object = {}'.format(goal))
        return goal.copy()
