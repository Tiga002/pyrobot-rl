import numpy as np
import os
import matplotlib.pyplot as plt

pwd = os.path.dirname(os.path.realpath(__file__))
gazebo_err_dir = os.path.abspath(os.path.join(pwd, os.pardir, 'using_gazebo_model', '5', 'gazebo_observations_rel_dist_5.txt'))
mujoco_err_dir = os.path.abspath(os.path.join(pwd, os.pardir, 'using_gazebo_model', '5', 'mujoco_observations_rel_dist_5.txt'))

gazebo_err_data = np.loadtxt(gazebo_err_dir)
mujoco_err_data = np.loadtxt(mujoco_err_dir)

x = np.arange(0,50) # timesteps


plt.suptitle('Relative Distance from goal Observation Comparison among Mujoco and Gazebo Environment using agent trained in Gazebo # 5')

plt.plot(x, gazebo_err_data)
plt.plot(x, mujoco_err_data)
plt.xlabel('Timesteps')
plt.ylabel('Meters')
plt.legend(['Gazebo', 'Mujoco'])
plt.show()
