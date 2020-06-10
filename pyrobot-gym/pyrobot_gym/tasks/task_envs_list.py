#!/usr/bin/env python
from gym.envs.registration import register
from gym import envs

"""
Helper Function for registering the custom task environment to the OpenAI Gym
"""

def RegisterOpenAI_Ros_Env(task_env, max_episode_steps=10000):
    """
    Registers all the ENVS supported in OpenAI ROS. This way we can load them
    with variable limits.
    Here is where you have to PLACE YOUR NEW TASK ENV, to be registered and accesible.
    return: False if the Task_Env wasnt registered, True if it was.
    """

    ###########################################################################
    # MovingCube Task-Robot Envs

    result = True

    # LocoBot Reach Task
    if task_env == "LocoBotReach-v2":
        print("Import Module")

        # Import the class that we registered so that it can be found afterwards in the Make
        #from pyrobot_gym.tasks.gazebo_reach import LocoBotGazeboReachEnv

        print("Importing register env")
        # We register the Class through the Gym system
        register(
            id = task_env,
            entry_point='pyrobot_gym.tasks:LocoBotGazeboReachEnv',
            kwargs=kwargs,
            max_episode_steps=50)
    else:
        result = False

    ###################################################3

    if result:
        # Check if the env was really registered
        supported_gym_envs = GetAllRegisteredGymEnvs()
        assert (task_env in supported_gym_envs), "The Task_Robot_ENV given is not Registered ==>" + str(task_env)

    return result

def GetAllRegisteredGymEnvs():
    """
    Returns a List of all the registered Envs in the system
    return EX: ['Copy-v0', 'RepeatCopy-v0', 'ReversedAddition-v0', ... ]
    """

    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    return env_ids
