from baselines.common import plot_util as pu
mujoco_results = pu.load_results('/home/developer/logs/her_pyrobot_push_mujoco/1e6_v1')
#results = pu.load_results('/home/developer/logs/her_pyrobot_reach/joint_100k_v4')
import matplotlib.pyplot as plt
import numpy as np

mujoco_r = mujoco_results[0]
mujoco_arr = np.array(mujoco_r.progress)
epoch = mujoco_arr[:, 0]
mujoco_test_success_rate = mujoco_arr[:, 7]
mujoco_train_success_rate = mujoco_arr[:, 9]
plt.suptitle('LocoBot Push Task Trained in MuJoCo with 1e6 timesteps(24 epoches, 800episodes per epoch)~')

plt.title('Training in Mujoco')
#plt.plot(epoch, pu.smooth(mujoco_train_success_rate, radius=10), label="mujoco_train")
#plt.plot(epoch, pu.smooth(mujoco_test_success_rate, radius=10), label="mujoco_test")
plt.plot(epoch, mujoco_train_success_rate, label="mujoco_train")
plt.plot(epoch, mujoco_test_success_rate, label="mujoco_test")
plt.xlabel('epoches')
plt.ylabel('success_rate')
plt.legend()
plt.show()
