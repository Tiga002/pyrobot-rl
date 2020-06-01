import os
from gym import utils
#from gym_pyrobot.envs.pyrobot_core_env import PyRobotCoreEnv
from pyrobot_gym.robots import LocoBotMujocoEnv
# Load the simulation environment XML
pwd = os.path.dirname(os.path.realpath(__file__))
#MODEL_XML_PATH = os.path.join(pwd, 'assets', 'locobot', 'reach.xml')
MODEL_XML_PATH = os.path.abspath(os.path.join(pwd, os.pardir, 'assets', 'tasks', 'reach.xml'))
# DEBUG:
print("path = {}".format(MODEL_XML_PATH))

class LocoBotMujocoReachEnv(LocoBotMujocoEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'joint_1': -1.22487753e-05,
            'joint_2': 6.71766300e-03,
            'joint_3': 7.30874027e-03,
            'joint_4': 3.80183559e-03,
            'joint_5': -5.06792684e-05
        }
        LocoBotMujocoEnv.__init__(
            self,MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type)
        utils.EzPickle.__init__(self)
