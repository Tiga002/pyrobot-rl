from pyrobot_gym.robots.locobot_mujoco_env import LocoBotMujocoEnv

try:
    from pyrobot_gym.robots.locobot_gazebo_env import LocoBotGazeboEnv
    from pyrobot_gym.robots.locobot_env import LocoBotEnv
except ModuleNotFoundError:
    LocoBotGazeboEnv = None
    LocoBotEnv = None
