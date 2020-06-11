from baselines.common import plot_util as pu
mujoco_results = pu.load_results('/home/developer/logs/her_pyrobot_reach/joint_10k_v9')
gazebo_results = pu.load_results('/home/developer/logs/her_pyrobot_reach_gazebo/joint_10k_v5')
#results = pu.load_results('/home/developer/logs/her_pyrobot_reach/joint_100k_v4')
import matplotlib.pyplot as plt
import numpy as np

mujoco_r = mujoco_results[0]
gazebo_r = gazebo_results[0]
mujoco_arr = np.array(mujoco_r.progress)
gazebo_arr = np.array(gazebo_r.progress)
epoch = mujoco_arr[:, 0]
mujoco_test_success_rate = mujoco_arr[:, 7]
gazebo_test_success_rate = gazebo_arr[:, 7]
mujoco_train_success_rate = mujoco_arr[:, 9]
gazebo_train_success_rate = gazebo_arr[:, 9]
plt.suptitle('LocoBot Reach Task Training in Mujoco vs Training in Gazebo ~')

plt.subplot(2,1,1)
plt.title('Training in Mujoco')
plt.plot(epoch, mujoco_train_success_rate, label="mujoco_train")
plt.plot(epoch, mujoco_test_success_rate, label="mujoco_test")
plt.xlabel('epoches')
plt.ylabel('success_rate')
plt.legend()

plt.subplot(2,1,2)
plt.title('Training in Gazebo')
plt.plot(epoch, gazebo_train_success_rate, label="gazebo_train")
plt.plot(epoch, gazebo_test_success_rate, label="gazebo_test")
plt.xlabel('epoches')
plt.ylabel('success_rate')
plt.legend()
plt.show()
