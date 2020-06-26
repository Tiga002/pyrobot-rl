from baselines.common import plot_util as pu
mujoco_results = pu.load_results('/home/developer/logs/her_pyrobot_push_mujoco/250k_v2')
#results = pu.load_results('/home/developer/logs/her_pyrobot_reach/joint_100k_v4')
import matplotlib.pyplot as plt
import numpy as np

mujoco_r = mujoco_results[0]
mujoco_arr = np.array(mujoco_r.progress)
epoch = mujoco_arr[:, 0]
mujoco_test_success_rate = mujoco_arr[:, 7]
mujoco_train_success_rate = mujoco_arr[:, 9]
plt.suptitle('LocoBot Push Task Trained in MuJoCo with 250k timesteps(100 epoches)~')

plt.title('Training in Mujoco')
plt.plot(epoch, pu.smooth(mujoco_train_success_rate, radius=10), label="mujoco_train")
plt.plot(epoch, pu.smooth(mujoco_test_success_rate, radius=10), label="mujoco_test")
plt.xlabel('epoches')
plt.ylabel('success_rate')
plt.legend()
#plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))
plt.show()
