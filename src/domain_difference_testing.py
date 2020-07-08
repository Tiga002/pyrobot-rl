import gym
import pyrobot_gym
import numpy as np
import math
import matplotlib.pyplot as plt
import time
"""
mujoco_observations_joint_1 = []
mujoco_observations_joint_1_eff_x = []
mujoco_observations_joint_1_eff_y = []
mujoco_observations_joint_1_eff_z = []

mujoco_observations_joint_2 = []
mujoco_observations_joint_2_eff_x = []
mujoco_observations_joint_2_eff_y = []
mujoco_observations_joint_2_eff_z = []

mujoco_observations_joint_3 = []
mujoco_observations_joint_3_eff_x = []
mujoco_observations_joint_3_eff_y = []
mujoco_observations_joint_3_eff_z = []

mujoco_observations_joint_4 = []
mujoco_observations_joint_4_eff_x = []
mujoco_observations_joint_4_eff_y = []
mujoco_observations_joint_4_eff_z = []

gazebo_observations_joint_1 = []
gazebo_observations_joint_1_eff_x = []
gazebo_observations_joint_1_eff_y = []
gazebo_observations_joint_1_eff_z = []

gazebo_observations_joint_2 = []
gazebo_observations_joint_2_eff_x = []
gazebo_observations_joint_2_eff_y = []
gazebo_observations_joint_2_eff_z = []

gazebo_observations_joint_3 = []
gazebo_observations_joint_3_eff_x = []
gazebo_observations_joint_3_eff_y = []
gazebo_observations_joint_3_eff_z = []

gazebo_observations_joint_4 = []
gazebo_observations_joint_4_eff_x = []
gazebo_observations_joint_4_eff_y = []
gazebo_observations_joint_4_eff_z = []

"""
real_observations_joint_1 = []
real_observations_joint_1_eff_x = []
real_observations_joint_1_eff_y = []
real_observations_joint_1_eff_z = []

real_observations_joint_2 = []
real_observations_joint_2_eff_x = []
real_observations_joint_2_eff_y = []
real_observations_joint_2_eff_z = []

real_observations_joint_3 = []
real_observations_joint_3_eff_x = []
real_observations_joint_3_eff_y = []
real_observations_joint_3_eff_z = []

real_observations_joint_4 = []
real_observations_joint_4_eff_x = []
real_observations_joint_4_eff_y = []
real_observations_joint_4_eff_z = []

"""
def main():

    env = gym.make('LocoBotReach-v1')
    env.reset()
    total_timesteps = 100
    freq = 2
    timestep = np.arange(total_timesteps)
    desired_joint_trag = np.sin(2*np.pi*freq*(timestep/total_timesteps))
    for t in range(total_timesteps):  #100 timesteps
        action = np.array([desired_joint_trag[t],0., 0, 0.])
        obs, reward, done, info = env.step(action)
        env.render()
        joint_position = obs['observation'][-5:][0]
        eff_position_x = obs['observation'][:3][0]
        eff_position_y = obs['observation'][:3][1]
        eff_position_z = obs['observation'][:3][2]
        mujoco_observations_joint_1.append(joint_position)
        mujoco_observations_joint_1_eff_x.append(eff_position_x)
        mujoco_observations_joint_1_eff_y.append(eff_position_y)
        mujoco_observations_joint_1_eff_z.append(eff_position_z)
    np.savetxt('Comparison/MuJoCo/joint_1.txt', mujoco_observations_joint_1)
    np.savetxt('Comparison/MuJoCo/joint_1_eff_x.txt', mujoco_observations_joint_1_eff_x)
    np.savetxt('Comparison/MuJoCo/joint_1_eff_y.txt', mujoco_observations_joint_1_eff_y)
    np.savetxt('Comparison/MuJoCo/joint_1_eff_z.txt', mujoco_observations_joint_1_eff_z)
    env.reset()
    for t in range(total_timesteps):  #100 timesteps
        action = np.array([0., desired_joint_trag[t]*0.7, 0, 0.])
        obs, reward, done, info = env.step(action)
        env.render()
        joint_position = obs['observation'][-5:][1]
        eff_position_x = obs['observation'][:3][0]
        eff_position_y = obs['observation'][:3][1]
        eff_position_z = obs['observation'][:3][2]
        mujoco_observations_joint_2.append(joint_position)
        mujoco_observations_joint_2_eff_x.append(eff_position_x)
        mujoco_observations_joint_2_eff_y.append(eff_position_y)
        mujoco_observations_joint_2_eff_z.append(eff_position_z)
    np.savetxt('Comparison/MuJoCo/joint_2.txt', mujoco_observations_joint_2)
    np.savetxt('Comparison/MuJoCo/joint_2_eff_x.txt', mujoco_observations_joint_2_eff_x)
    np.savetxt('Comparison/MuJoCo/joint_2_eff_y.txt', mujoco_observations_joint_2_eff_y)
    np.savetxt('Comparison/MuJoCo/joint_2_eff_z.txt', mujoco_observations_joint_2_eff_z)
    env.reset()
    for t in range(total_timesteps):  #100 timesteps
        action = np.array([0., 0., desired_joint_trag[t], 0.])
        obs, reward, done, info = env.step(action)
        env.render()
        joint_position = obs['observation'][-5:][2]
        eff_position_x = obs['observation'][:3][0]
        eff_position_y = obs['observation'][:3][1]
        eff_position_z = obs['observation'][:3][2]
        mujoco_observations_joint_3.append(joint_position)
        mujoco_observations_joint_3_eff_x.append(eff_position_x)
        mujoco_observations_joint_3_eff_y.append(eff_position_y)
        mujoco_observations_joint_3_eff_z.append(eff_position_z)
    np.savetxt('Comparison/MuJoCo/joint_3.txt', mujoco_observations_joint_3)
    np.savetxt('Comparison/MuJoCo/joint_3_eff_x.txt', mujoco_observations_joint_3_eff_x)
    np.savetxt('Comparison/MuJoCo/joint_3_eff_y.txt', mujoco_observations_joint_3_eff_y)
    np.savetxt('Comparison/MuJoCo/joint_3_eff_z.txt', mujoco_observations_joint_3_eff_z)
    env.reset()
    for t in range(total_timesteps):  #100 timesteps
        action = np.array([0., 0., 0., desired_joint_trag[t]])
        obs, reward, done, info = env.step(action)
        env.render()
        joint_position = obs['observation'][-5:][3]
        eff_position_x = obs['observation'][:3][0]
        eff_position_y = obs['observation'][:3][1]
        eff_position_z = obs['observation'][:3][2]
        mujoco_observations_joint_4.append(joint_position)
        mujoco_observations_joint_4_eff_x.append(eff_position_x)
        mujoco_observations_joint_4_eff_y.append(eff_position_y)
        mujoco_observations_joint_4_eff_z.append(eff_position_z)
    np.savetxt('Comparison/MuJoCo/joint_4.txt', mujoco_observations_joint_4)
    np.savetxt('Comparison/MuJoCo/joint_4_eff_x.txt', mujoco_observations_joint_4_eff_x)
    np.savetxt('Comparison/MuJoCo/joint_4_eff_y.txt', mujoco_observations_joint_4_eff_y)
    np.savetxt('Comparison/MuJoCo/joint_4_eff_z.txt', mujoco_observations_joint_4_eff_z)
    env.close()
#############################################################################
    env = gym.make('LocoBotReach-v2')
    env.reset()
    total_timesteps = 100
    freq = 2
    timestep = np.arange(total_timesteps)
    desired_joint_trag = np.sin(2*np.pi*freq*(timestep/total_timesteps))
    for t in range(total_timesteps):  #100 timesteps
        action = np.array([desired_joint_trag[t],0., 0, 0.])
        obs, reward, done, info = env.step(action)
        #env.render()
        joint_position = obs['observation'][-5:][0]
        eff_position_x = obs['observation'][:3][0]
        eff_position_y = obs['observation'][:3][1]
        eff_position_z = obs['observation'][:3][2]
        gazebo_observations_joint_1.append(joint_position)
        gazebo_observations_joint_1_eff_x.append(eff_position_x)
        gazebo_observations_joint_1_eff_y.append(eff_position_y)
        gazebo_observations_joint_1_eff_z.append(eff_position_z)
    np.savetxt('Comparison/Gazebo/gazebo_joint_1.txt', gazebo_observations_joint_1)
    np.savetxt('Comparison/Gazebo/gazebo_joint_1_eff_x.txt', gazebo_observations_joint_1_eff_x)
    np.savetxt('Comparison/Gazebo/gazebo_joint_1_eff_y.txt', gazebo_observations_joint_1_eff_y)
    np.savetxt('Comparison/Gazebo/gazebo_joint_1_eff_z.txt', gazebo_observations_joint_1_eff_z)
    env.reset()
    for t in range(total_timesteps):  #100 timesteps
        action = np.array([0., desired_joint_trag[t]*0.7, 0, 0.])
        obs, reward, done, info = env.step(action)
        #env.render()
        joint_position = obs['observation'][-5:][1]
        eff_position_x = obs['observation'][:3][0]
        eff_position_y = obs['observation'][:3][1]
        eff_position_z = obs['observation'][:3][2]
        gazebo_observations_joint_2.append(joint_position)
        gazebo_observations_joint_2_eff_x.append(eff_position_x)
        gazebo_observations_joint_2_eff_y.append(eff_position_y)
        gazebo_observations_joint_2_eff_z.append(eff_position_z)
    np.savetxt('Comparison/Gazebo/gazebo_joint_2.txt', gazebo_observations_joint_2)
    np.savetxt('Comparison/Gazebo/gazebo_joint_2_eff_x.txt', gazebo_observations_joint_2_eff_x)
    np.savetxt('Comparison/Gazebo/gazebo_joint_2_eff_y.txt', gazebo_observations_joint_2_eff_y)
    np.savetxt('Comparison/Gazebo/gazebo_joint_2_eff_z.txt', gazebo_observations_joint_2_eff_z)
    env.reset()
    for t in range(total_timesteps):  #100 timesteps
        action = np.array([0., 0., desired_joint_trag[t], 0.])
        obs, reward, done, info = env.step(action)
        #env.render()
        joint_position = obs['observation'][-5:][2]
        eff_position_x = obs['observation'][:3][0]
        eff_position_y = obs['observation'][:3][1]
        eff_position_z = obs['observation'][:3][2]
        gazebo_observations_joint_3.append(joint_position)
        gazebo_observations_joint_3_eff_x.append(eff_position_x)
        gazebo_observations_joint_3_eff_y.append(eff_position_y)
        gazebo_observations_joint_3_eff_z.append(eff_position_z)
    np.savetxt('Comparison/Gazebo/gazebo_joint_3.txt', gazebo_observations_joint_3)
    np.savetxt('Comparison/Gazebo/gazebo_joint_3_eff_x.txt', gazebo_observations_joint_3_eff_x)
    np.savetxt('Comparison/Gazebo/gazebo_joint_3_eff_y.txt', gazebo_observations_joint_3_eff_y)
    np.savetxt('Comparison/Gazebo/gazebo_joint_3_eff_z.txt', gazebo_observations_joint_3_eff_z)
    env.reset()
    for t in range(total_timesteps):  #100 timesteps
        action = np.array([0., 0., 0., desired_joint_trag[t]])
        obs, reward, done, info = env.step(action)
        #env.render()
        joint_position = obs['observation'][-5:][3]
        eff_position_x = obs['observation'][:3][0]
        eff_position_y = obs['observation'][:3][1]
        eff_position_z = obs['observation'][:3][2]
        gazebo_observations_joint_4.append(joint_position)
        gazebo_observations_joint_4_eff_x.append(eff_position_x)
        gazebo_observations_joint_4_eff_y.append(eff_position_y)
        gazebo_observations_joint_4_eff_z.append(eff_position_z)
    np.savetxt('Comparison/Gazebo/gazebo_joint_4.txt', gazebo_observations_joint_4)
    np.savetxt('Comparison/Gazebo/gazebo_joint_4_eff_x.txt', gazebo_observations_joint_4_eff_x)
    np.savetxt('Comparison/Gazebo/gazebo_joint_4_eff_y.txt', gazebo_observations_joint_4_eff_y)
    np.savetxt('Comparison/Gazebo/gazebo_joint_4_eff_z.txt', gazebo_observations_joint_4_eff_z)
    env.close()
#############################################################################
"""
def main():
    env = gym.make('LocoBotReach-v3')
    env.reset()
    total_timesteps = 100
    freq = 2
    timestep = np.arange(total_timesteps)
    desired_joint_trag = np.sin(2*np.pi*freq*(timestep/total_timesteps))
    for t in range(total_timesteps):  #100 timesteps
        action = np.array([desired_joint_trag[t],0., 0, 0.])
        obs, reward, done, info = env.step(action)
        #env.render()
        joint_position = obs['observation'][-5:][0]
        eff_position_x = obs['observation'][:3][0]
        eff_position_y = obs['observation'][:3][1]
        eff_position_z = obs['observation'][:3][2]
        real_observations_joint_1.append(joint_position)
        real_observations_joint_1_eff_x.append(eff_position_x)
        real_observations_joint_1_eff_y.append(eff_position_y)
        real_observations_joint_1_eff_z.append(eff_position_z)
    np.savetxt('Comparison/Real/real_joint_1.txt', real_observations_joint_1)
    np.savetxt('Comparison/Real/real_joint_1_eff_x.txt', real_observations_joint_1_eff_x)
    np.savetxt('Comparison/Real/real_joint_1_eff_y.txt', real_observations_joint_1_eff_y)
    np.savetxt('Comparison/Real/real_joint_1_eff_z.txt', real_observations_joint_1_eff_z)
    env.reset()
    for t in range(total_timesteps):  #100 timesteps
        action = np.array([0., desired_joint_trag[t]*0.7, 0, 0.])
        obs, reward, done, info = env.step(action)
        #env.render()
        joint_position = obs['observation'][-5:][1]
        eff_position_x = obs['observation'][:3][0]
        eff_position_y = obs['observation'][:3][1]
        eff_position_z = obs['observation'][:3][2]
        real_observations_joint_2.append(joint_position)
        real_observations_joint_2_eff_x.append(eff_position_x)
        real_observations_joint_2_eff_y.append(eff_position_y)
        real_observations_joint_2_eff_z.append(eff_position_z)
    np.savetxt('Comparison/Real/real_joint_2.txt', real_observations_joint_2)
    np.savetxt('Comparison/Real/real_joint_2_eff_x.txt', real_observations_joint_2_eff_x)
    np.savetxt('Comparison/Real/real_joint_2_eff_y.txt', real_observations_joint_2_eff_y)
    np.savetxt('Comparison/Real/real_joint_2_eff_z.txt', real_observations_joint_2_eff_z)
    env.reset()
    for t in range(total_timesteps):  #100 timesteps
        action = np.array([0., 0., desired_joint_trag[t], 0.])
        obs, reward, done, info = env.step(action)
        #env.render()
        joint_position = obs['observation'][-5:][2]
        eff_position_x = obs['observation'][:3][0]
        eff_position_y = obs['observation'][:3][1]
        eff_position_z = obs['observation'][:3][2]
        real_observations_joint_3.append(joint_position)
        real_observations_joint_3_eff_x.append(eff_position_x)
        real_observations_joint_3_eff_y.append(eff_position_y)
        real_observations_joint_3_eff_z.append(eff_position_z)
    np.savetxt('Comparison/Real/real_joint_3.txt', real_observations_joint_3)
    np.savetxt('Comparison/Real/real_joint_3_eff_x.txt', real_observations_joint_3_eff_x)
    np.savetxt('Comparison/Real/real_joint_3_eff_y.txt', real_observations_joint_3_eff_y)
    np.savetxt('Comparison/Real/real_joint_3_eff_z.txt', real_observations_joint_3_eff_z)
    env.reset()
    for t in range(total_timesteps):  #100 timesteps
        action = np.array([0., 0., 0., desired_joint_trag[t]])
        obs, reward, done, info = env.step(action)
        #env.render()
        joint_position = obs['observation'][-5:][3]
        eff_position_x = obs['observation'][:3][0]
        eff_position_y = obs['observation'][:3][1]
        eff_position_z = obs['observation'][:3][2]
        real_observations_joint_4.append(joint_position)
        real_observations_joint_4_eff_x.append(eff_position_x)
        real_observations_joint_4_eff_y.append(eff_position_y)
        real_observations_joint_4_eff_z.append(eff_position_z)
    np.savetxt('Comparison/Real/real_joint_4.txt', real_observations_joint_4)
    np.savetxt('Comparison/Real/real_joint_4_eff_x.txt', real_observations_joint_4_eff_x)
    np.savetxt('Comparison/Real/real_joint_4_eff_y.txt', real_observations_joint_4_eff_y)
    np.savetxt('Comparison/Real/real_joint_4_eff_z.txt', real_observations_joint_4_eff_z)
    env.close()

"""
    fig, ax = plt.subplots(2,2)
    fig.suptitle("Desired Joint Position vs Actual Joint Positions")

    ax[0,0].set_title('Joint_1')
    ax[0,0].plot(timestep, desired_joint_trag, 'b')
    ax[0,0].plot(timestep, observations_joint_1, 'r')
    ax[0,0].plot(timestep, gazebo_observations_joint_1, 'g')
    ax[0,0].plot(timestep, real_observations_joint_1, 'c')
    ax[0,0].set_ylabel('Joint Angles (radian)')
    ax[0,0].set_xlabel('Timesteps')
    ax[0,0].legend(['Desired', 'Mujoco', 'Gazebo', 'Real'])

    ax[0,1].set_title('Joint_2')
    ax[0,1].plot(timestep, desired_joint_trag*0.65, 'b')
    ax[0,1].plot(timestep, observations_joint_2, 'r')
    ax[0,1].plot(timestep, gazebo_observations_joint_2, 'g')
    ax[0,1].plot(timestep, real_observations_joint_2, 'c')
    ax[0,1].set_ylabel('Joint Angles (radian)')
    ax[0,1].set_xlabel('Timesteps')
    ax[0,1].legend(['Desired', 'Mujoco', 'Gazebo', 'Real'])

    ax[1,0].set_title('Joint_3')
    ax[1,0].plot(timestep, desired_joint_trag, 'b')
    ax[1,0].plot(timestep, observations_joint_3, 'r')
    ax[1,0].plot(timestep, gazebo_observations_joint_3, 'g')
    ax[1,0].plot(timestep, real_observations_joint_3, 'c')
    ax[1,0].set_ylabel('Joint Angles (radian)')
    ax[1,0].set_xlabel('Timesteps')
    ax[1,0].legend(['Desired', 'Mujoco', 'Gazebo', 'Real'])

    ax[1,1].set_title('Joint_4')
    ax[1,1].plot(timestep, desired_joint_trag, 'b')
    ax[1,1].plot(timestep, observations_joint_4, 'r')
    ax[1,1].plot(timestep, gazebo_observations_joint_4, 'g')
    ax[1,1].plot(timestep, real_observations_joint_4, 'c')
    ax[1,1].set_ylabel('Joint Angles (radian)')
    ax[1,1].set_xlabel('Timesteps')
    ax[1,1].legend(['Desired', 'Mujoco', 'Gazebo', 'Real'])

    plt.subplots_adjust(left=0.125,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.35)

    plt.show()
"""
if __name__ == "__main__":
    main()
