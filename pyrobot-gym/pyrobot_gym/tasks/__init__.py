# Task: Reach
# Task: Push
# Task: Push
from pyrobot_gym.tasks.mujoco_push import LocoBotMujocoPushEnv, \
    LocoBotMujocoPushEnv
from pyrobot_gym.tasks.mujoco_reach import LocoBotMujocoReachEnv, \
    LocoBotMujocoReachEnv
from pyrobot_gym.tasks.task_commons import LoadYamlFileParamsTest, \
    LoadYamlFileParamsTest
from pyrobot_gym.tasks.task_envs_list import RegisterOpenAI_Ros_Env, \
    GetAllRegisteredGymEnvs, RegisterOpenAI_Ros_Env, GetAllRegisteredGymEnvs

try:
    from pyrobot_gym.tasks.gazebo_reach import LocoBotGazeboReachEnv
    from pyrobot_gym.tasks.gazebo_push import LocoBotGazeboPushEnv
    from pyrobot_gym.tasks.reach import LocoBotReachEnv, LocoBotReachEnv
    from pyrobot_gym.tasks.gazebo_push import LocoBotGazeboPushEnv
except:
    LocoBotGazeboReachEnv = None
    LocoBotGazeboPushEnv = None
    LocoBotReachEnv = None
    LocoBotGazeboPushEnv = None
