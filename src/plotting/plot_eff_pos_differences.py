import numpy as np
import os
import matplotlib.pyplot as plt

pwd = os.path.dirname(os.path.realpath(__file__))
gazebo_eff_pos_dir = os.path.abspath(os.path.join(pwd, os.pardir, 'using_gazebo_model', '5', 'gazebo_observations_eff_pos_5.txt'))
mujoco_eff_pos_dir = os.path.abspath(os.path.join(pwd, os.pardir, 'using_gazebo_model', '5', 'mujoco_observations_eff_pos_5.txt'))

gazebo_eff_pos_data = np.loadtxt(gazebo_eff_pos_dir)
mujoco_eff_pos_data = np.loadtxt(mujoco_eff_pos_dir)

gazebo_eff_x = gazebo_eff_pos_data[:, 0]
gazebo_eff_y = gazebo_eff_pos_data[:, 1]
gazebo_eff_z = gazebo_eff_pos_data[:, 2]

mujoco_eff_x = mujoco_eff_pos_data[:, 0]
mujoco_eff_y = mujoco_eff_pos_data[:, 1]
mujoco_eff_z = mujoco_eff_pos_data[:, 2]

x = np.arange(0,50) # timesteps

fig, ax = plt.subplots(2, 2)
fig.suptitle('End Effector Observation Comparison among Mujoco and Gazebo Environment using agent trained in Gazebo # 5')

#plt.subplot(4,1,1)
ax[0,0].set_title('End-Effector X Position')
ax[0,0].plot(x,gazebo_eff_x)
ax[0,0].plot(x, mujoco_eff_x)
ax[0,0].set_ylabel('Meters')
ax[0,0].set_xlabel('Timesteps')
ax[0,0].legend(['Gazebo', 'Mujoco'])

ax[0,1].set_title('End-Effector Y Position')
ax[0,1].plot(x,gazebo_eff_y)
ax[0,1].plot(x, mujoco_eff_y)
ax[0,1].set_ylabel('Meters')
ax[0,1].set_xlabel('Timesteps')
ax[0,1].legend(['Gazebo', 'Mujoco'])

ax[1,0].set_title('End-Effector Z Position')
ax[1,0].plot(x,gazebo_eff_z)
ax[1,0].plot(x, mujoco_eff_z)
ax[1,0].set_ylabel('Meters')
ax[1,0].set_xlabel('Timesteps')
ax[1,0].legend(['Gazebo', 'Mujoco'])


plt.subplots_adjust(left=0.125,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.35)

plt.show()
