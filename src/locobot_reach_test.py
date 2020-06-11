#!/usr/bin/env python
import gym
#import gym_pyrobot
import pyrobot_gym
import numpy as np

""" Sample PyRobot Manipulation in Mujoco Environment """
actions = []
observation = []
in0fos = []

def main():
    env = gym.make('LocoBotReach-v1')
    #env = gym.make('FetchReach-v1')
    numItr = 100
    initStateSpace = "random"
    env.reset()
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()
        #print("ITERATION NUMBER ", len(actions))
        env.render()
        #print('env.action_space = {}'.format(env.action_space))i
        action = env.action_space.sample()
        #action = np.array([0.02, -0.9, 0.023, 0., 0.2])
        #print('===== Action = {}'.format(action))
        obs = env.step(action)

#        reachToGoal(env, obs)

def reachToGoal(env, lastObs):
    goal = lastObs['desired_goal']
    ## DEBUG:
    print('Desired Goal Position = {}'.format(goal))
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    timeStep = 0
    episodeObs.append(lastObs)
    distance = np.linalg.norm(goal, axis=-1)
    while distance >= 0.05: # and timeStep <= env._max_episode_steps:
        print("==================================")
        env.render()
        action = [0, 0, 0, 0]
        for i in range(len(goal)):
            action[i] = goal[i]*6
            print('action[i] = {}'.format(action[i]))
            print('goal[i] = {}'.format(goal[i]))

        action[len(action)-1] = 0 #remain close

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        achieved_goal = obsDataNew['achieved_goal']
        print("achieved_goal = {}".format(achieved_goal))
        eff_pos = obsDataNew['observation'][:3]
        print("eff_pos = {}".format(eff_pos))
        desired_goal = obsDataNew['desired_goal']
        print("desired_goal = {}".format(desired_goal))
        distance_vector = achieved_goal - desired_goal
        print('distance_vector = {}'.format(distance_vector))
        distance = np.linalg.norm(distance_vector, axis=-1)
        print('distance = {}'.format(distance))

    while True:
        env.render()
        action = [0, 0, 0, 0]
        action[len(action)-1] = 0 # keep the gripper closed

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        if timeStep >= 10000: break

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)

if __name__ == "__main__":
    main()









"""
env = gym.make('pyrobot-reach-v0')
#env = gym.make('FetchReach-v1')
env.reset()

for _ in range(1000):
    env.render()

    env.step(env.action_space.sample())
env.close()
"""
