#!/usr/bin/env python
import time

import gym
import numpy as np


def main():
    import pyrobot_gym
    pyrobot_gym
    env = gym.make('LocoBotPush-v1')
    # env = gym.make('FetchReach-v1')
    numItr = 50
    env.reset()

    print("Reset!")
    for aid in range(4):
        for pos_neg in range(-1, 2, 2):  # Positive and nagative
            for i in range(numItr):
                obs, r, d, i = env.step([
                    float(aid == 0) * 0.1 * pos_neg,
                    float(aid == 1) * 0.1 * pos_neg,
                    float(aid == 2) * 0.1 * pos_neg,
                    float(aid == 3) * 0.1 * pos_neg,
                ])
                print(f"Obs {obs}, Reward {r}, Done {d}, Info {i}")
                env.render()
                time.sleep(0.1)
                if d:
                    env.reset()


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
    while distance >= 0.05:  # and timeStep <= env._max_episode_steps:
        print("==================================")
        env.render()
        action = [0, 0, 0, 0]
        for i in range(len(goal)):
            action[i] = goal[i] * 6
            print('action[i] = {}'.format(action[i]))
            print('goal[i] = {}'.format(goal[i]))

        action[len(action) - 1] = 0  # remain close

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
        action[len(action) - 1] = 0  # keep the gripper closed

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        if timeStep >= 10000:
            break


if __name__ == "__main__":
    main()
