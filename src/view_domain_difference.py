import numpy as np
import os
import matplotlib.pyplot as plt

pwd = os.path.dirname(os.path.realpath(__file__))
mujoco_joint_1_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_1.txt'))
mujoco_joint_1_eff_x_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_1_eff_x.txt'))
mujoco_joint_1_eff_y_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_1_eff_y.txt'))
mujoco_joint_1_eff_z_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_1_eff_z.txt'))
mujoco_joint_1 = np.loadtxt(mujoco_joint_1_dir)
mujoco_joint_1_eff_x = np.loadtxt(mujoco_joint_1_eff_x_dir)
mujoco_joint_1_eff_y = np.loadtxt(mujoco_joint_1_eff_y_dir)
mujoco_joint_1_eff_z = np.loadtxt(mujoco_joint_1_eff_z_dir)

mujoco_joint_2_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_2.txt'))
mujoco_joint_2_eff_x_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_2_eff_x.txt'))
mujoco_joint_2_eff_y_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_2_eff_y.txt'))
mujoco_joint_2_eff_z_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_2_eff_z.txt'))
mujoco_joint_2 = np.loadtxt(mujoco_joint_2_dir)
mujoco_joint_2_eff_x = np.loadtxt(mujoco_joint_2_eff_x_dir)
mujoco_joint_2_eff_y = np.loadtxt(mujoco_joint_2_eff_y_dir)
mujoco_joint_2_eff_z = np.loadtxt(mujoco_joint_2_eff_z_dir)

mujoco_joint_3_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_3.txt'))
mujoco_joint_3_eff_x_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_3_eff_x.txt'))
mujoco_joint_3_eff_y_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_3_eff_y.txt'))
mujoco_joint_3_eff_z_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_3_eff_z.txt'))
mujoco_joint_3 = np.loadtxt(mujoco_joint_3_dir)
mujoco_joint_3_eff_x = np.loadtxt(mujoco_joint_3_eff_x_dir)
mujoco_joint_3_eff_y = np.loadtxt(mujoco_joint_3_eff_y_dir)
mujoco_joint_3_eff_z = np.loadtxt(mujoco_joint_3_eff_z_dir)

mujoco_joint_4_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_4.txt'))
mujoco_joint_4_eff_x_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_4_eff_x.txt'))
mujoco_joint_4_eff_y_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_4_eff_y.txt'))
mujoco_joint_4_eff_z_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'MuJoCo', 'joint_4_eff_z.txt'))
mujoco_joint_4 = np.loadtxt(mujoco_joint_4_dir)
mujoco_joint_4_eff_x = np.loadtxt(mujoco_joint_4_eff_x_dir)
mujoco_joint_4_eff_y = np.loadtxt(mujoco_joint_4_eff_y_dir)
mujoco_joint_4_eff_z = np.loadtxt(mujoco_joint_4_eff_z_dir)
########################################################################
gazebo_joint_1_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_1.txt'))
gazebo_joint_1_eff_x_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_1_eff_x.txt'))
gazebo_joint_1_eff_y_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_1_eff_y.txt'))
gazebo_joint_1_eff_z_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_1_eff_z.txt'))
gazebo_joint_1 = np.loadtxt(gazebo_joint_1_dir)
gazebo_joint_1_eff_x = np.loadtxt(gazebo_joint_1_eff_x_dir)
gazebo_joint_1_eff_y = np.loadtxt(gazebo_joint_1_eff_y_dir)
gazebo_joint_1_eff_z = np.loadtxt(gazebo_joint_1_eff_z_dir)

gazebo_joint_2_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_2.txt'))
gazebo_joint_2_eff_x_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_2_eff_x.txt'))
gazebo_joint_2_eff_y_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_2_eff_y.txt'))
gazebo_joint_2_eff_z_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_2_eff_z.txt'))
gazebo_joint_2 = np.loadtxt(gazebo_joint_2_dir)
gazebo_joint_2_eff_x = np.loadtxt(gazebo_joint_2_eff_x_dir)
gazebo_joint_2_eff_y = np.loadtxt(gazebo_joint_2_eff_y_dir)
gazebo_joint_2_eff_z = np.loadtxt(gazebo_joint_2_eff_z_dir)

gazebo_joint_3_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_3.txt'))
gazebo_joint_3_eff_x_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_3_eff_x.txt'))
gazebo_joint_3_eff_y_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_3_eff_y.txt'))
gazebo_joint_3_eff_z_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_3_eff_z.txt'))
gazebo_joint_3 = np.loadtxt(gazebo_joint_3_dir)
gazebo_joint_3_eff_x = np.loadtxt(gazebo_joint_3_eff_x_dir)
gazebo_joint_3_eff_y = np.loadtxt(gazebo_joint_3_eff_y_dir)
gazebo_joint_3_eff_z = np.loadtxt(gazebo_joint_3_eff_z_dir)

gazebo_joint_4_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_4.txt'))
gazebo_joint_4_eff_x_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_4_eff_x.txt'))
gazebo_joint_4_eff_y_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_4_eff_y.txt'))
gazebo_joint_4_eff_z_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Gazebo', 'gazebo_joint_4_eff_z.txt'))
gazebo_joint_4 = np.loadtxt(gazebo_joint_4_dir)
gazebo_joint_4_eff_x = np.loadtxt(gazebo_joint_4_eff_x_dir)
gazebo_joint_4_eff_y = np.loadtxt(gazebo_joint_4_eff_y_dir)
gazebo_joint_4_eff_z = np.loadtxt(gazebo_joint_4_eff_z_dir)
#########################################################################
real_joint_1_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_1.txt'))
real_joint_1_eff_x_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_1_eff_x.txt'))
real_joint_1_eff_y_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_1_eff_y.txt'))
real_joint_1_eff_z_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_1_eff_z.txt'))
real_joint_1 = np.loadtxt(real_joint_1_dir)
real_joint_1_eff_x = np.loadtxt(real_joint_1_eff_x_dir)
real_joint_1_eff_y = np.loadtxt(real_joint_1_eff_y_dir)
real_joint_1_eff_z = np.loadtxt(real_joint_1_eff_z_dir)

real_joint_2_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_2.txt'))
real_joint_2_eff_x_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_2_eff_x.txt'))
real_joint_2_eff_y_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_2_eff_y.txt'))
real_joint_2_eff_z_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_2_eff_z.txt'))
real_joint_2 = np.loadtxt(real_joint_2_dir)
real_joint_2_eff_x = np.loadtxt(real_joint_2_eff_x_dir)
real_joint_2_eff_y = np.loadtxt(real_joint_2_eff_y_dir)
real_joint_2_eff_z = np.loadtxt(real_joint_2_eff_z_dir)

real_joint_3_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_3.txt'))
real_joint_3_eff_x_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_3_eff_x.txt'))
real_joint_3_eff_y_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_3_eff_y.txt'))
real_joint_3_eff_z_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_3_eff_z.txt'))
real_joint_3 = np.loadtxt(real_joint_3_dir)
real_joint_3_eff_x = np.loadtxt(real_joint_3_eff_x_dir)
real_joint_3_eff_y = np.loadtxt(real_joint_3_eff_y_dir)
real_joint_3_eff_z = np.loadtxt(real_joint_3_eff_z_dir)

real_joint_4_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_4.txt'))
real_joint_4_eff_x_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_4_eff_x.txt'))
real_joint_4_eff_y_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_4_eff_y.txt'))
real_joint_4_eff_z_dir = os.path.abspath(os.path.join(pwd, 'Comparison', 'Real', 'real_joint_4_eff_z.txt'))
real_joint_4 = np.loadtxt(real_joint_4_dir)
real_joint_4_eff_x = np.loadtxt(real_joint_4_eff_x_dir)
real_joint_4_eff_y = np.loadtxt(real_joint_4_eff_y_dir)
real_joint_4_eff_z = np.loadtxt(real_joint_4_eff_z_dir)


fig, ax = plt.subplots(6,2)
fig.suptitle("End-Effector Positions Comparison [Joint 1 & 2]")
#fig.suptitle("MuJoCo Joint Position vs Real Joint Positions")

total_timesteps = 100
freq = 2
timestep = np.arange(total_timesteps)
desired_joint_trag = np.sin(2*np.pi*freq*(timestep/total_timesteps))
ax[0,0].set_title('Eff X Position')
#ax[0,0].plot(timestep, desired_joint_trag, 'b')
ax[0,0].plot(timestep,  mujoco_joint_1_eff_x, 'r')
ax[0,0].plot(timestep, gazebo_joint_1_eff_x, 'g')
ax[0,0].plot(timestep, real_joint_1_eff_x, 'c')
ax[0,0].set_ylabel('Position (meter')
#ax[0,0].set_xlabel('Timesteps')
ax[0,0].legend(['Mujoco', 'Gazebo', 'Real'])
#ax[0,0].legend(['MuJoCo', 'Real'])

ax[1,0].set_title('Eff Y Position')
#ax[0,1].plot(timestep, desired_joint_trag*0.7, 'b')
ax[1,0].plot(timestep,  mujoco_joint_1_eff_y, 'r')
ax[1,0].plot(timestep, gazebo_joint_1_eff_y, 'g')
ax[1,0].plot(timestep, real_joint_1_eff_y, 'c')
ax[1,0].set_ylabel('Position (meter')
#ax[1,0].set_xlabel('Timesteps')
ax[1,0].legend(['Mujoco', 'Gazebo', 'Real'])
#ax[1,0].legend(['MuJoCo', 'Real'])

ax[2,0].set_title('Eff Z Position')
#ax[1,0].plot(timestep, desired_joint_trag, 'b')
ax[2,0].plot(timestep,  mujoco_joint_1_eff_z, 'r')
ax[2,0].plot(timestep, gazebo_joint_1_eff_z, 'g')
ax[2,0].plot(timestep, real_joint_1_eff_z, 'c')
ax[2,0].set_ylabel('Position (meter')
#ax[2,0].set_xlabel('Timesteps')
ax[2,0].legend(['Mujoco', 'Gazebo', 'Real'])
#ax[2,0].legend(['MuJoCo', 'Real'])

ax[1,1].set_title('Joint_1')
ax[1,1].plot(timestep, desired_joint_trag, 'b')
ax[1,1].plot(timestep, mujoco_joint_1, 'r')
ax[1,1].plot(timestep, gazebo_joint_1, 'g')
ax[1,1].plot(timestep, real_joint_1, 'c')
ax[1,1].set_ylabel('Joint Angles (radian)')
#ax[1,1].set_xlabel('Timesteps')
ax[1,1].legend(['Desired', 'Mujoco', 'Gazebo', 'Real'])
#ax[1,1].legend(['MuJoCo', 'Real'])


ax[3,0].set_title('Eff X Position')
#ax[0,0].plot(timestep, desired_joint_trag, 'b')
ax[3,0].plot(timestep,  mujoco_joint_2_eff_x, 'r')
ax[3,0].plot(timestep, gazebo_joint_2_eff_x, 'g')
ax[3,0].plot(timestep, real_joint_2_eff_x, 'c')
ax[3,0].set_ylabel('Position (meter')
#ax[3,0].set_xlabel('Timesteps')
ax[3,0].legend(['Mujoco', 'Gazebo', 'Real'])
#ax[0,0].legend(['MuJoCo', 'Real'])

ax[4,0].set_title('Eff Y Position')
#ax[0,1].plot(timestep, desired_joint_trag*0.7, 'b')
ax[4,0].plot(timestep,  mujoco_joint_2_eff_y, 'r')
ax[4,0].plot(timestep, gazebo_joint_2_eff_y, 'g')
ax[4,0].plot(timestep, real_joint_2_eff_y, 'c')
ax[4,0].set_ylabel('Position (meter')
#ax[4,0].set_xlabel('Timesteps')
ax[4,0].legend(['Mujoco', 'Gazebo', 'Real'])
#ax[1,0].legend(['MuJoCo', 'Real'])

ax[5,0].set_title('Eff Z Position')
#ax[1,0].plot(timestep, desired_joint_trag, 'b')
ax[5,0].plot(timestep,  mujoco_joint_2_eff_z, 'r')
ax[5,0].plot(timestep, gazebo_joint_2_eff_z, 'g')
ax[5,0].plot(timestep, real_joint_2_eff_z, 'c')
ax[5,0].set_ylabel('Position (meter')
ax[5,0].set_xlabel('Timesteps')
ax[5,0].legend(['Mujoco', 'Gazebo', 'Real'])
#ax2,0].legend(['MuJoCo', 'Real'])

ax[4,1].set_title('Joint_2')
ax[4,1].plot(timestep, desired_joint_trag*0.7, 'b')
ax[4,1].plot(timestep, mujoco_joint_2, 'r')
ax[4,1].plot(timestep, gazebo_joint_2, 'g')
ax[4,1].plot(timestep, real_joint_2, 'c')
ax[4,1].set_ylabel('Joint Angles (radian)')
ax[4,1].set_xlabel('Timesteps')
ax[4,1].legend(['Desired', 'Mujoco', 'Gazebo', 'Real'])
#ax[1,1].legend(['MuJoCo', 'Real'])



plt.subplots_adjust(left=0.125,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.55)

plt.show()
