import numpy as np

data = np.load('gazebo_5.npz')
"""
# Print the data
lst = data.files
for item in lst:
    print(item)
    print(data[item])
"""

"""Save the data to csv
for key, value in data.items():
    np.savetxt("mujoco_data_2500.csv", value)
"""
lst = data.files  # lst=['acs', 'obs', 'info'
actions = data['acs']
actions = np.squeeze(actions)
#print(actions.shape)
obs = data['obs']
obs = np.squeeze(obs)
#observations = np.squeeze(observations)
#print(observations.shape)
arr = [0.0]*9
observations = np.array([arr])
#print(observations.shape)
for i in range(50):
    observation = obs[i].get("observation")
    #print(observation.shape)
    observations = np.append(observations, observation, axis=0)
observations = np.delete(observations, 0, 0)
eff_pos = observations[:, 0:3]
relative_dis_ee_goal = observations[:, 3]
joint_pos = observations[:, 4:]
#observations = np.squeeze(observations)
#print(observations.shape)
infos = data['info']
infos = np.squeeze(infos)
print(infos.shape)
infos = np.squeeze(infos)
print("=== Episode Information ===")
print("goal = {}".format(obs[0].get("desired_goal")))
print(infos[-1].get('is_success'))


np.savetxt('gazebo_actions_5.txt', actions)
np.savetxt('gazebo_observations_eff_pos_5.txt', eff_pos)
np.savetxt('gazebo_observations_rel_dist_5.txt', relative_dis_ee_goal)
np.savetxt('gazebo_observations_joint_pos_5.txt', joint_pos)
f = open("gazebo_episode_info_5.txt", "w")
f.write("=== Episode Information === \n")
f.write("goal = {} \n".format(obs[0].get("desired_goal")))
f.write("done = {}".format(infos[-1].get('is_success')))
f.close()
#np.savetxt('mujoco_info', np.array([infos[-1].get('is_success')]))
#np.savetxt('mujoco_infos.txt', infos)
