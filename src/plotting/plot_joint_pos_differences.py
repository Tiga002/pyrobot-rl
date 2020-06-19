import numpy as np
import os
import matplotlib.pyplot as plt

pwd = os.path.dirname(os.path.realpath(__file__))
gazebo_joint_pos_dir = os.path.abspath(os.path.join(pwd, os.pardir, 'using_gazebo_model', '5', 'gazebo_observations_joint_pos_5.txt'))
mujoco_joint_pos_dir = os.path.abspath(os.path.join(pwd, os.pardir, 'using_gazebo_model', '5', 'mujoco_observations_joint_pos_5.txt'))

gazebo_joint_pos_data = np.loadtxt(gazebo_joint_pos_dir)
mujoco_joint_pos_data = np.loadtxt(mujoco_joint_pos_dir)

gazebo_joint_1 = gazebo_joint_pos_data[:, 0]
gazebo_joint_2 = gazebo_joint_pos_data[:, 1]
gazebo_joint_3 = gazebo_joint_pos_data[:, 2]
gazebo_joint_4 = gazebo_joint_pos_data[:, 3]

mujoco_joint_1 = mujoco_joint_pos_data[:, 0]
mujoco_joint_2 = mujoco_joint_pos_data[:, 1]
mujoco_joint_3 = mujoco_joint_pos_data[:, 2]
mujoco_joint_4 = mujoco_joint_pos_data[:, 3]

x = np.arange(0,50) # timesteps

fig, ax = plt.subplots(2, 2)
fig.suptitle('Joint Angles Observation Comparison among Mujoco and Gazebo Environment using agent trained in Gazebo # 5')

#plt.subplot(4,1,1)
ax[0,0].set_title('Joint_1')
ax[0,0].plot(x, gazebo_joint_1)
ax[0,0].plot(x, mujoco_joint_1)
ax[0,0].set_ylabel('Joint Angles (radian)')
ax[0,0].set_xlabel('Timesteps')
ax[0,0].legend(['Gazebo', 'Mujoco'])

ax[0,1].set_title('Joint_2')
ax[0,1].plot(x, gazebo_joint_2)
ax[0,1].plot(x, mujoco_joint_2)
ax[0,1].set_ylabel('Joint Angles (radian)')
ax[0,1].set_xlabel('Timesteps')
ax[0,1].legend(['Gazebo', 'Mujoco'])

ax[1,0].set_title('Joint_3')
ax[1,0].plot(x, gazebo_joint_3)
ax[1,0].plot(x, mujoco_joint_3)
ax[1,0].set_ylabel('Joint Angles (radian)')
ax[1,0].set_xlabel('Timesteps')
ax[1,0].legend(['Gazebo', 'Mujoco'])

ax[1,1].set_title('Joint_4')
ax[1,1].plot(x, gazebo_joint_4)
ax[1,1].plot(x, mujoco_joint_4)
ax[1,1].set_ylabel('Joint Angles (radian)')
ax[1,1].set_xlabel('Timesteps')
ax[1,1].legend(['Gazebo', 'Mujoco'])

plt.subplots_adjust(left=0.125,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.35)

plt.show()
