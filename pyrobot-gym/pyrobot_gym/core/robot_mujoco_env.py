import os
import copy
import numpy as np
import gym
from gym.utils import seeding
from random import seed as random_seed
from random import randint

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

class RobotMujocoEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps,randomize_action_timesteps):
        #print("[RobotMujocoEnv] START init RobotMujocoEnv")
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=1)
        self.viewer = None
        self._viewers = {}
        self.randomize_action_timesteps = randomize_action_timesteps
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()

        #print('[DEBUG@RobotEnv] initial_qpos = {}'.format(initial_qpos))
        #self._env_setup(initial_qpos=initial_qpos)

        """ For testing
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.initial_state.qpos[5] = -1.22487753e-05
        self.initial_state.qpos[6] = 6.71766300e-03
        self.initial_state.qpos[7] = 7.30874027e-03
        self.initial_state.qpos[8] = 3.80183559e-03
        self.initial_state.qpos[9] = -5.06792684e-05
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        """
        #print("[RobotMujocoEnv] END init RobotMujocoEnv")

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------
    def do_simulation(self, action, n_frames):
        self._set_action(action)
        for _ in range(n_frames):
            self.sim.step()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random_seed(123)
        return [seed]

    def step(self, action):
        # Normalize the action within (-1,1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        #print('action = {}'.format(action))
        n_frames = 20
        if self.randomize_action_timesteps == True:
            n_frames = n_frames + randint(0,10)
            #print('action timesteps = {}'.format(n_frames))
        self.do_simulation(action, n_frames)
        #self._step_callback()
        obs = self._get_obs()  # TODO: add real robot implementation
        done = False
        info = {
                'is_success': self._is_success(obs['achieved_goal'], self.goal)
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        #print('[STEP] Rewards = {}'.format(reward))
        #print('[STEP] info = {}'.format(info))
        return obs, reward, done, info

    def reset(self):
        #print('[RESET]')
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        obs = self._get_obs()
        #print('End [RESET]')
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    ## Domain Randomization helper functions
    def get_id(self, obj_name, prop_name):
        if prop_name == 'body_mass':
            id = self.sim.model.body_name2id(obj_name)
        elif prop_name == 'dof_damping':
            id = self.sim.model.joint_name2id(obj_name)
        elif prop_name == 'geom_mass':
            id = self.sim.model.geom_name2id(obj_name)
        elif prop_name == 'actuator_gainprm':
            id = self.sim.model.actuator_name2id(obj_name)
        else:
            id = None
        return id

    def set_property(self, obj_name, prop_name, prop_value):
        id = self.get_id(obj_name, prop_name)
        prop_all = getattr(self.sim.model, prop_name)
        prop_all[id] = prop_value
        prop_all = getattr(self.sim.model, prop_name)

    def get_property(self, obj_name, prop_name):
        id = self.get_id(obj_name, prop_name)
        prop_all = getattr(self.sim.model, prop_name)
        prop_val = prop_all[id]
        return prop_val
    # Extension methods
    # ---------------------------
    def _reset_sim(self):
        """
        Reset the Simulation by reset all the joints to initial positions

        Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        # Set to the startup position and sample a goal
        self._set_init_pose()
        return True

    def _set_init_pose(self):
        """
        Sets the Robot in its startup initial Pose
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
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
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
